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

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#define CPU_ONLY

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
// #include "caffe/util/format.hpp"
// #include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "lmdb.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::shared_ptr;
const char SEPARATOR = ' ';

/* List command-line flags */
DEFINE_bool(shuffle, false,
            "Randomly shuffle the order of images and their labels");
DEFINE_bool(sync_db, false,
            "Sync the output database with the list if labels and images");

/* Read a file if <image path><sep><label> pairs, where sep = SEPARATOR. */
std::vector<std::pair<std::string, int> > read_images(const std::string& path) {
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

// Create new DB
shared_ptr<LMDB> open_or_create_db(const std::string& source, bool sync_db) {
    shared_ptr<LMDB> db(new LMDB());
    if (! sync_db) {
        db->Open(source, LMDB::NEW);
    }

    return db;
}

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

    // Read LABEL_FILE (= the file maps image paths to labels).
    std::vector<std::pair<std::string, int> > image_label_lines;
    image_label_lines = read_images(argv[2]);

    if (FLAGS_shuffle) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        shuffle(image_label_lines.begin(), image_label_lines.end());
    }

    shared_ptr<LMDB> db = open_or_create_db(argv[3], FLAGS_sync_db);

    return 0;
}
