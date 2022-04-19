
// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <iostream>
#include <random>
#include <gtest/gtest.h>
#include <math.h>
#include <memory>
#include <string>
#include <utility>

#include "unittest/utils.h"

#include "knowhere/common/Config.h"
#include "knowhere/index/vector_index/IndexHNSW.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/common/Exception.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class HNSWTest : public DataGen {
 public:
    void
    SetUp(int dim, int nb, std::string metricType, /*base Params*/
          int M, int efConstruction, /*index Params*/
          int ef, int topK, int nq /*search Params*/) {
        std::cout << "dim = " << dim << ", nb = " << nb << ", metric_type = " << metricType << std::endl;
        std::cout << "index_type = HNSW, M = " << M << ", efConstruction = " << efConstruction << std::endl;
        std::cout << "topK = " << topK << ", nq = " << nq << ", ef = " << ef << std::endl;
        DataGen::Generate(dim, nb, nq);
        index_ = std::make_shared<milvus::knowhere::IndexHNSW>();
        conf = milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, dim},        {milvus::knowhere::meta::TOPK, topK},
            {milvus::knowhere::IndexParams::M, M},   {milvus::knowhere::IndexParams::efConstruction, efConstruction},
            {milvus::knowhere::IndexParams::ef, ef}, {milvus::knowhere::Metric::TYPE, metricType},
        };
    }

 public:
    milvus::knowhere::Config conf;
    std::shared_ptr<milvus::knowhere::IndexHNSW> index_ = nullptr;
};

int
main() {
    auto dim = 128, nb = 100000; auto metric_type = milvus::knowhere::Metric::L2;
    auto M = 12, efConstruction = 150;
    auto ef = 50, topK = 50, nq = 10;

    auto test = HNSWTest();
    test.SetUp(dim, nb, metric_type, M, efConstruction, ef, topK, nq);

    assert(!test.xb.empty());

    test.index_->Train(test.base_dataset, test.conf);
    test.index_->AddWithoutIds(test.base_dataset, test.conf);
    EXPECT_EQ(test.index_->Count(), test.nb);
    EXPECT_EQ(test.index_->Dim(), test.dim);

    // Serialize and Load before Query
    milvus::knowhere::BinarySet bs = test.index_->Serialize(test.conf);

    int64_t data_dim = test.base_dataset->Get<int64_t>(milvus::knowhere::meta::DIM);
    int64_t rows = test.base_dataset->Get<int64_t>(milvus::knowhere::meta::ROWS);
    auto raw_data = test.base_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR);
    milvus::knowhere::BinaryPtr bptr = std::make_shared<milvus::knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = data_dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    test.index_->Load(bs);

    auto result = test.index_->Query(test.query_dataset, test.conf, nullptr);
    AssertAnns(result, test.nq, test.k);

    // case: k > nb
    const int64_t new_rows = 6;
    test.base_dataset->Set(milvus::knowhere::meta::ROWS, new_rows);
    test.index_->Train(test.base_dataset, test.conf);
    test.index_->AddWithoutIds(test.base_dataset, test.conf);
    auto result2 = test.index_->Query(test.query_dataset, test.conf, nullptr);
    auto res_ids = result2->Get<int64_t*>(milvus::knowhere::meta::IDS);
    for (int64_t i = 0; i < test.nq; i++) {
        for (int64_t j = new_rows; j < test.k; j++) {
            EXPECT_EQ(res_ids[i * test.k + j], -1);
        }
    }
}