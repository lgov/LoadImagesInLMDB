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

#include "lmdb.hpp"

#include <sys/stat.h>
#include <stdexcept>

#include <lmdb.h>

void LMDB::Open(const std::string& source, LMDB::Mode mode) {
    if (int error = mdb_env_create(&mdb_env_)) {
        mdb_env_close(mdb_env_);
        throw std::runtime_error("Failure opening LMDB environment");
    }

    if (mode == LMDB::NEW) {
        if (mkdir(source.c_str(), 0744)) {
            throw std::runtime_error("mkdir failed");
        }
    }
}

void LMDB::Close() {
    if (mdb_env_ != NULL) {
        mdb_dbi_close(mdb_env_, mdb_dbi_);
        mdb_env_close(mdb_env_);
        mdb_env_ = NULL;
    }
}
