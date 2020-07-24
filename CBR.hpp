
template<class Tup>
struct SchemaCallBackBinReduce {
        typedef  std::get<0>(Tup) T1;
        typedef  std::get<1>(Tup) T2;
        typedef  std::get<2>(Tup) T3;
        typedef  std::get<3>(Tup) T4;
        typedef  std::get<4>(Tup) T5;
        typedef  std::get<5>(Tup) T6;
        typedef  std::get<6>(Tup) T7;
        typedef  std::get<7>(Tup) T8;
        typedef  std::get<8>(Tup) T9;
      
        static void call(std::function<void(const RuntimeConfig&,const CSRWrapper&,BackwardGData<T3,T4>*)> body)
        {
           if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
              // cpu::FallbackCallBackwardBinaryReduce<Tup>::call(rtcfg, graph, gdata);
              cpu::FallbackCallBackwardBinaryReduce<T1,T2,T3,T4,T5,T6,T7,T8>(rtcfg, graph, gdata);

             } else {
               body(rtcfg,graph,gdata);
             }         
        }
};

using ListOfType = std::tuple<kDLCPU,binary_op::kGradLhs,int32_t,double,SelectSrc, SelectNone,BinaryUseLhs<double>, ReduceSum<kDLCPU,double >; 

SchemaCallBackBinReduce<ListOfType>::call([](const RuntimeConfig& rtcfg,const CSRWrapper& graph,BackwardGData<int32_t, double>* gdata ){ 

     auto csr = graph.GetOutCSRMatrix();
     cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data, gdata->x_length);
});



template <>
void CallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, double,
                              SelectSrc, SelectNone,
                              BinaryUseLhs<double>, ReduceSum<kDLCPU, double>>(
    const RuntimeConfig& rtcfg,
    const CSRWrapper& graph,
    BackwardGData<int32_t, double>* gdata) {
    if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
        cpu::FallbackCallBackwardBinaryReduce<kDLCPU, binary_op::kGradLhs, int32_t, double,
                                              SelectSrc, SelectNone,
                                              BinaryUseLhs<double>, ReduceSum<kDLCPU, double>>
            (rtcfg, graph, gdata);
    } else {
        auto csr = graph.GetOutCSRMatrix();
        cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data,
                        gdata->x_length);
    }
}


