/* Copyright 2017 Lieven Govaerts
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Based on the convert_image_set.cpp program included in the tools section
// of the Caffe source distribution.

#include <iostream>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <thread>

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>

#include "gflags/gflags.h"
#include "glog/logging.h"

#define CPU_ONLY

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
// #include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "lmdb.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::vector;
using boost::shared_ptr;
using boost::scoped_ptr;
const char SEPARATOR = ' ';

/* List command-line flags */
DEFINE_bool(shuffle, false,
            "Randomly shuffle the order of images and their labels");
DEFINE_bool(sync_db, false,
            "Sync the output database with the list if labels and images");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");

/* Read a file if <image path><sep><label> pairs, where sep = SEPARATOR. */
std::vector<std::pair<std::string, int> > read_image_labels(const std::string& path) {
    std::ifstream infile(path);
    std::vector<std::pair<std::string, int> > lines;
    std::string line;
    size_t pos;
    int label;

    while (std::getline(infile, line)) {
        pos = line.find_last_of(SEPARATOR);
        label = atoi(line.substr(pos + 1).c_str());
        lines.push_back(std::make_pair(line.substr(0, pos), label));
    }

    return lines;
}

// Open an existing database of create a new one.
LMDB* open_or_create_db(const std::string& source, bool sync_db) {
    LMDB* db(new LMDB());
    if (sync_db) {
        db->Open(source, LMDB::WRITE);
    } else {
        db->Open(source, LMDB::NEW);
    }

    return db;
}

shared_ptr<Datum> load_image(const std::string& source, int label, const int resize_width, const int resize_height) {
    bool status;
    shared_ptr<Datum> datum(new Datum());

    bool is_color = true;
    string encode_type = "";
    status = ReadImageToDatum(source, label, resize_height, resize_width, is_color,
                              encode_type, datum.get());
    return datum;
}

const std::string& path_join(const std::string &path1, const std::string &path2) {

    boost::filesystem::path full_path (path1);
    boost::filesystem::path file (path2);
    full_path /= file;

    return full_path.string();
}

/* A lock-free queue, used to pass the read <key> <datum> pairs from the 
   reader thread to the writer thread */
boost::lockfree::spsc_queue< pair<std::string, caffe::Datum> > queue(128);
boost::atomic<bool> done_writing (false);

class ReaderThread {
public:
    ReaderThread(vector<pair<std::string, int> > image_label_lines,
                 std::string root_folder, int resize_width, int resize_height) :
                    image_label_lines_(image_label_lines),
                    root_folder_(root_folder), resize_width_(resize_width),
                    resize_height_(resize_height) { }

    void operator()() {
        LOG(INFO) << "Starting to import " << image_label_lines_.size() << " files.";

        for (int line_id = 0; line_id < image_label_lines_.size(); ++line_id) {
            std::string image_path = image_label_lines_[line_id].first;
            int image_label = image_label_lines_[line_id].second;
            std::string full_path = path_join(root_folder_, image_path);

            //        std::cout << "Image - label: " << std::to_string(image_label) << "\n";
            //        std::cout << "Path: " << full_path << "\n";

            std::string key = caffe::format_int(line_id, 8);
            shared_ptr<caffe::Datum> datum = load_image(full_path, image_label, resize_width_, resize_height_);

            // push key, Datum in the queue
            while (!queue.push(std::make_pair(key, *datum)))
                ;

            // DEBUG
            if (line_id % 100 == 0) {
                LOG(INFO) << "Processed " << line_id << " files.";
            }

            if ((line_id + 1) % 10000 == 0) {
                break;
            }
        }
    }
private:
    std::vector<std::pair<std::string, int> > image_label_lines_;
    std::string root_folder_;
    int resize_height_;
    int resize_width_;
};

class WriterThread {
public:
    WriterThread(std::string db_name): db_name_(db_name) { }

    void operator()() {
        db_ = open_or_create_db(db_name_, FLAGS_sync_db);

        size_t id = 0;
        scoped_ptr<LMDBTransaction> txn(db_->NewTransaction());

        while (! done_writing) {
            store_all_on_queue(txn, id);
        }

        store_all_on_queue(txn, id);
    }

    void store_all_on_queue(scoped_ptr<LMDBTransaction>& txn, size_t& id) {
        pair<std::string, caffe::Datum> value;
        std::string key = "";

        while (queue.pop(value)) {
            bool success = db_->StoreDatum(txn.get(), key, &value.second);

            if ((id > 0) && (id % 100 == 0)) {
                txn->Commit();
                txn.reset(db_->NewTransaction());
                LOG(INFO) << "Processed " << id << " files.";
            }

//            LOG(INFO) << " available on queue: " << queue.read_available();
        }
    }
    virtual ~WriterThread() { if (db_) delete db_; db_ = NULL; }
private:
    std::string db_name_;
    LMDB* db_;
};

// this should take a const char * argv[], but ParseCommandLineFlags wants
// non-const
int main(int argc, char * argv[]) {
    // Print output to stderr (while still logging)
//    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    ::google::InitGoogleLogging(argv[0]);

    gflags::SetUsageMessage("Imports a set of images in a new or existing LMDB database,\n"
                            "used as input for Caffe.\n"
                            "Usage:\n"
                            "    load_images_in_ldmb [FLAGS] IMAGES_FOLDER/ LABEL_FILE DB_NAME\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 4) {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "load_images_in_ldmb");
        return 1;
    }

    std::string root_folder(argv[1]);
    std::string image_labels_file(argv[2]);
    std::string db_name(argv[3]);

    // Read LABEL_FILE (= the file maps image paths to labels).
    vector<pair<std::string, int> > image_label_lines;
    image_label_lines = read_image_labels(image_labels_file);

    if (FLAGS_shuffle) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        shuffle(image_label_lines.begin(), image_label_lines.end());
    }
    int resize_height = std::max<int>(0, FLAGS_resize_height);
    int resize_width  = std::max<int>(0, FLAGS_resize_width);

    ReaderThread rt(image_label_lines, root_folder, resize_width, resize_height);
    std::thread reader(rt);
    WriterThread wt(db_name);
    std::thread writer(wt);

    // First finish reading all images
    reader.join();

    // Then finish storing them all in the database
    writer.join();

    return 0;
}
