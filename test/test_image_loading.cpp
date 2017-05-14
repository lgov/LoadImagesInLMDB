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

#define BOOST_TEST_MODULE TestImageLoading
#include <boost/test/unit_test.hpp>

#define CPU_ONLY
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <string>
#include "boost/shared_ptr.hpp"

using boost::shared_ptr;

static const std::string images_folder = "test/images/";

BOOST_AUTO_TEST_CASE(load_jpg_and_resize)
{
    bool success;
    shared_ptr<caffe::Datum> datum(new caffe::Datum());

    bool is_color = true;
    std::string encode_type = "";
    std::string source = images_folder + "640px-Volga_Estate_Anvers.jpg";

    success = ReadImageToDatum(source, 123456, 240, 260, is_color,
                              encode_type, datum.get());

    BOOST_CHECK( success );
    BOOST_CHECK( datum->has_channels() == true );
    BOOST_CHECK_EQUAL( datum->channels(), 3 );
    BOOST_CHECK_EQUAL( datum->height(), 240 );
    BOOST_CHECK_EQUAL( datum->width(), 260 );
    BOOST_CHECK_EQUAL( datum->label(), 123456 );
    BOOST_CHECK( datum->encoded() == false );
    const std::string& data = datum->data();
    BOOST_CHECK_EQUAL( data.size(), 260 * 240 * 3 );
}


