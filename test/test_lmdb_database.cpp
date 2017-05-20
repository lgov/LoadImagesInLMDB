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

#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

#define CPU_ONLY
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using boost::shared_ptr;
using boost::scoped_ptr;

#include "lmdb.hpp"

static const std::string databases_folder = "test/test_working/";
static const std::string images_folder = "test/images/";

BOOST_AUTO_TEST_CASE(create_database)
{
    std::string db_path = databases_folder + "test_create_database";
    boost::filesystem::remove_all(db_path);

    shared_ptr<LMDB> db(new LMDB());
    db->Open(db_path, LMDB::NEW);

    BOOST_CHECK( db );
}

BOOST_AUTO_TEST_CASE(import_image_in_database)
{
    bool success;
    scoped_ptr<caffe::Datum> datum(new caffe::Datum());

    std::string db_path = databases_folder + "test_import_image_in_database";
    boost::filesystem::remove_all(db_path);

    std::string image = images_folder + "640px-Volga_Estate_Anvers.jpg";

    bool is_color = true; std::string encode_type = "";
    success = ReadImageToDatum(image, 123456, 256, 256, is_color,
                               encode_type, datum.get());

    shared_ptr<LMDB> db(new LMDB());
    db->Open(db_path, LMDB::NEW);
    BOOST_CHECK( db );

    std::string key = caffe::format_int(1, 8);

    success = db->StoreDatum(key, datum.get());
    BOOST_CHECK( success );

    db->Close();
}

shared_ptr<LMDB> open_test_database(std::string db_pathj) {

    std::string db_full_path = databases_folder + db_pathj;

    /* Don't remove the root folder or anything. */
    if (db_full_path.length() < 10) {
        return shared_ptr<LMDB>();
    }
    boost::filesystem::remove_all(db_full_path);
    shared_ptr<LMDB> db(new LMDB());
    db->Open(db_full_path, LMDB::NEW);

    return db;
}

BOOST_AUTO_TEST_CASE(import_multiple_images_in_database)
{
    bool success;
    bool is_color = true;
    std::string encode_type = "";
    std::string key;

    /* Cleanup database folder and create a new one */
    shared_ptr<LMDB> db = open_test_database("test_read_image_in_database");

    /* Load the first image */
    scoped_ptr<caffe::Datum> datum(new caffe::Datum());
    std::string image1 = images_folder + "640px-Volga_Estate_Anvers.jpg";
    success = ReadImageToDatum(image1, 234567, 256, 256, is_color, encode_type, datum.get());
    key = caffe::format_int(1, 8);
    success = db->StoreDatum(key, datum.get());

    BOOST_CHECK_EQUAL(db->NrOfEntries(), 1);

    datum.reset(new caffe::Datum());
    std::string image2 = images_folder + "1200px-Phở_bò,_Cầu_Giấy,_Hà_Nội.jpg";
    success = ReadImageToDatum(image1, 234567, 256, 256, is_color, encode_type, datum.get());
    key = caffe::format_int(2, 8);
    success = db->StoreDatum(key, datum.get());

    BOOST_CHECK_EQUAL(db->NrOfEntries(), 2);

    db->Close();
    /*    ...     */
}

BOOST_AUTO_TEST_CASE(import_multiple_images_in_database_single_transaction)
{
    bool success;
    bool is_color = true;
    std::string encode_type = "";
    std::string key;

    /* Cleanup database folder and create a new one */
    shared_ptr<LMDB> db = open_test_database("test_read_image_in_database");

    /* Initiate transaction */
    scoped_ptr<LMDBTransaction> txn(db->NewTransaction());

    /* Load the first image */
    scoped_ptr<caffe::Datum> datum(new caffe::Datum());
    std::string image1 = images_folder + "640px-Volga_Estate_Anvers.jpg";
    success = ReadImageToDatum(image1, 234567, 256, 256, is_color, encode_type, datum.get());
    key = caffe::format_int(1, 8);
    success = db->StoreDatum(txn.get(), key, datum.get());

    /* Check that this transaction isn't committed yet. */
    BOOST_CHECK_EQUAL(db->NrOfEntries(), 0);

    datum.reset(new caffe::Datum());
    std::string image2 = images_folder + "1200px-Phở_bò,_Cầu_Giấy,_Hà_Nội.jpg";
    success = ReadImageToDatum(image1, 234567, 256, 256, is_color, encode_type, datum.get());
    key = caffe::format_int(2, 8);
    success = db->StoreDatum(txn.get(), key, datum.get());

    /* Commit, so now 2 entries become visible for everyone. */
    txn->Commit();
    BOOST_CHECK_EQUAL(db->NrOfEntries(), 2);

    db->Close();
    /*    ...     */
}
