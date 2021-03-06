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

// Based on the db_lmdb code included in the Caffe source distribution.

#include "lmdb.hpp"

#include <sys/stat.h>
#include <stdexcept>
#include <boost/scoped_ptr.hpp>

#include <lmdb.h>

#include "glog/logging.h"


using namespace caffe;  // NOLINT(build/namespaces)
using boost::scoped_ptr;

void LMDB::Open(const std::string& source, LMDB::Mode mode) {

    // Create the target folder for a new database
    if (mode == LMDB::NEW) {
        if (mkdir(source.c_str(), 0744)) {
            throw std::runtime_error("mkdir failed");
        }
    }

    if (int error = mdb_env_create(&mdb_env_)) {
        mdb_env_close(mdb_env_);
        throw std::runtime_error("Failure creating LMDB environment");
    }

    if (int error = mdb_env_open(mdb_env_, source.c_str(), 0, 0664)) {
        mdb_env_close(mdb_env_);
        throw std::runtime_error("Failure opening LMDB environment");
    }

    // db connection created
}

void LMDB::Close() {
    if (mdb_env_ != NULL) {
        mdb_dbi_close(mdb_env_, mdb_dbi_);
        mdb_env_close(mdb_env_);
        mdb_env_ = NULL;
    }
}

bool LMDB::StoreDatum(const std::string &key, const caffe::Datum * datum) {

    // Create transaction, add and commit.
    scoped_ptr<LMDBTransaction> txn(NewTransaction());
    if (! StoreDatum(txn.get(), key, datum)) {
        return false;
    }
    return txn->Commit();
}

bool LMDB::StoreString(LMDBTransaction *txn, const std::string &key, const std::string &datum_str) {

    return txn->Put(key, datum_str);
}


bool LMDB::StoreDatum(LMDBTransaction *txn, const std::string &key, const Datum* datum) {

    std::string out;
    if (datum->SerializeToString(&out)) {
        return StoreString(txn, key, out);
    }
    return false;
}

LMDBTransaction* LMDB::NewTransaction() {
    MDB_dbi mdb_dbi;
    MDB_txn *mdb_txn;

    // Initialize MDB variables
    if (mdb_txn_begin(mdb_env_, NULL /* no parent */, 0 /* rw */, &mdb_txn)) {
        return NULL;
    }
    if (mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi)) {
        return NULL;
    }

    return new LMDBTransaction(mdb_txn, mdb_dbi);
}


size_t LMDB::NrOfEntries() {
    MDB_stat stat;

    if (! mdb_env_stat(mdb_env_, &stat)) {
        return stat.ms_entries;
    }

    return SIZE_MAX;
}

/******************************************************************************/
/* LMDBTransaction                                                            */
/*                                                                            */
/******************************************************************************/
bool LMDBTransaction::Put(const std::string& key, const std::string& value) {

    MDB_val mdb_key, mdb_data;

    mdb_key.mv_size = key.size();
    mdb_key.mv_data = const_cast<char*>(key.data());
    mdb_data.mv_size = value.size();
    mdb_data.mv_data = const_cast<char*>(value.data());

    int put_rc = mdb_put(mdb_txn_, mdb_dbi_, &mdb_key, &mdb_data, 0);
    if (! put_rc) {
        return true;
    }
    LOG(ERROR) << "Txn Put failed. Aborting!" << put_rc;
    if (put_rc == MDB_MAP_FULL) {
        // Maximum map size exceeded, resize is needed.
        // We let the caller handle that.

        // TODO: exception
        return false;
    }
    return false;
}

bool LMDBTransaction::Commit() {
    MDB_env *env = mdb_txn_env(mdb_txn_);

    // Commit the transaction
    int commit_rc = mdb_txn_commit(mdb_txn_);
    if (! commit_rc) {
        return true;
    }
    // Cleanup after successful commit
    mdb_dbi_close(env, mdb_dbi_);

    return true;
}

bool LMDBTransaction::CommitAndDoubleMapSize() {
    if (! Commit()) {
        return false;
    }

    // Transaction is out of the way, now double the map size.
    struct MDB_envinfo current_info;
    MDB_env *env = mdb_txn_env(mdb_txn_);

    int rc = mdb_env_info(env, &current_info);
    if (rc) {
        return false;
    }
    size_t new_size = current_info.me_mapsize * 2;
    rc = mdb_env_set_mapsize(env, new_size);
    if (rc) {
        return false;
    }
    return true;
}


