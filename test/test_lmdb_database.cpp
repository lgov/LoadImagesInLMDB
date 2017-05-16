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
#include "boost/shared_ptr.hpp"

#define CPU_ONLY
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using boost::shared_ptr;

#include "lmdb.hpp"

static const std::string databases_folder = "test/test_working/";
static const std::string images_folder = "test/images/";

BOOST_AUTO_TEST_CASE(create_database)
{
    std::string source = databases_folder + "test_create_database";
    boost::filesystem::remove_all(source);

    shared_ptr<LMDB> db(new LMDB());
    db->Open(source, LMDB::NEW);

    BOOST_CHECK( db );
}

BOOST_AUTO_TEST_CASE(import_image_in_database)
{
    bool success;
    shared_ptr<caffe::Datum> datum(new caffe::Datum());

    std::string source = databases_folder + "test_import_image_in_database";
    std::string image = images_folder + "640px-Volga_Estate_Anvers.jpg";

    bool is_color = true; std::string encode_type = "";
    success = ReadImageToDatum(image, 123456, 256, 256, is_color,
                               encode_type, datum.get());

    shared_ptr<LMDB> db(new LMDB());
    db->Open(source, LMDB::NEW);
    BOOST_CHECK( db );

    std::string key = caffe::format_int(1, 8);

    success = db->StoreDatum(key, datum);
    BOOST_CHECK( success );

    db->Close();
}
