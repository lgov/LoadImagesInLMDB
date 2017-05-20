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

#ifndef lmdb_h
#define lmdb_h

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

using boost::shared_ptr;

#include <lmdb.h>
#include "caffe/proto/caffe.pb.h"

class LMDBTransaction {

public:
    LMDBTransaction(MDB_txn* mdb_txn, MDB_dbi mdb_dbi): mdb_txn_(mdb_txn), mdb_dbi_(mdb_dbi) { }
    bool Put(const std::string& key, const std::string& value);
    bool Commit();

private:
    MDB_txn* mdb_txn_;
    MDB_dbi mdb_dbi_;
};

class LMDB {

public:
    enum Mode { READ, WRITE, NEW };

    LMDB() : mdb_env_(NULL) { }
    virtual ~LMDB() { Close(); }
    void Open(const std::string& source, Mode mode);
    void Close();
    bool StoreDatum(LMDBTransaction *txn, const std::string &key, const caffe::Datum *  Datum);
    bool StoreDatum(const std::string &key, const caffe::Datum * Datum);
    LMDBTransaction* NewTransaction();
    size_t NrOfEntries();

private:
    MDB_env* mdb_env_;
    MDB_dbi mdb_dbi_;
};

#endif /* lmdb_h */
