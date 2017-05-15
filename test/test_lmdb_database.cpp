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

using boost::shared_ptr;

#include "lmdb.hpp"

static const std::string databases_folder = "test/test_working/";

BOOST_AUTO_TEST_CASE(create_database)
{
    std::string source = databases_folder + "test_create_database";

    shared_ptr<LMDB> db(new LMDB());
    db->Open(source, LMDB::NEW);

    BOOST_CHECK( db );
}

