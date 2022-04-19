
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

#include <gtest/gtest.h>
#include "knowhere/common/Config.h"
#include "knowhere/index/vector_index/IndexHNSW.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include <iostream>
#include <random>
#include "knowhere/common/Exception.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class HNSWTest : public DataGen, public TestWithParam<std::string> {
 protected:
    void
    SetUp() override {
        auto dim = 128, nb = 1000000; auto metric_type = milvus::knowhere::Metric::L2;
        auto M = 12, efConstruction = 150;
        auto ef = 50, topK = 50, nq = 100;
        std::cout << "dim = " << dim << ", nb = " << nb << ", metric_type = " << metric_type << std::endl;
        std::cout << "index_type = " << GetParam() << ", M = " << M << ", efConstruction = " << efConstruction << std::endl;
        std::cout << "topK = " << topK << ", nq = " << nq << ", ef = " << ef << std::endl;

        IndexType = GetParam();
        Generate(dim, nb, nq);
        index_ = std::make_shared<milvus::knowhere::IndexHNSW>();
        conf = milvus::knowhere::Config{
            {milvus::knowhere::meta::DIM, dim},        {milvus::knowhere::meta::TOPK, topK},
            {milvus::knowhere::IndexParams::M, M},   {milvus::knowhere::IndexParams::efConstruction, efConstruction},
            {milvus::knowhere::IndexParams::ef, ef}, {milvus::knowhere::Metric::TYPE, metric_type},
        };
    }

 protected:
    milvus::knowhere::Config conf;
    std::shared_ptr<milvus::knowhere::IndexHNSW> index_ = nullptr;
    std::string IndexType;
};

INSTANTIATE_TEST_CASE_P(HNSWParameters, HNSWTest, Values("HNSW"));

TEST_P(HNSWTest, PerfHNSW) {
#ifdef _DEBUG
    std::cout << "Build Debug Version" << std::endl;
#else
    std::cout << "Build Release Version" << std::endl;
#endif
    std::cout << "PID:" << ::getpid() << std::endl;
    assert(!xb.empty());

    index_->Train(base_dataset, conf);
    index_->AddWithoutIds(base_dataset, conf);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    // Serialize and Load before Query
    milvus::knowhere::BinarySet bs = index_->Serialize(conf);

    int64_t dim = base_dataset->Get<int64_t>(milvus::knowhere::meta::DIM);
    int64_t rows = base_dataset->Get<int64_t>(milvus::knowhere::meta::ROWS);
    auto raw_data = base_dataset->Get<const void*>(milvus::knowhere::meta::TENSOR);
    milvus::knowhere::BinaryPtr bptr = std::make_shared<milvus::knowhere::Binary>();
    bptr->data = std::shared_ptr<uint8_t[]>((uint8_t*)raw_data, [&](uint8_t*) {});
    bptr->size = dim * rows * sizeof(float);
    bs.Append(RAW_DATA, bptr);

    index_->Load(bs);

    int query_times = 1000000;
    std::cout << "start query..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < query_times; i++) {
        auto result = index_->Query(query_dataset, conf, nullptr);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "PerfHNSW done, query_times = " << query_times << ", threads_avg_duration = " << duration.count()/1000000.0/*to seconds*//query_times << " s" << std::endl;
}
