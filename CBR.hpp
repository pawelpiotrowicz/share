

template<class Tup>
struct SchemaCallBackBinReduce {
        typedef typename std::tuple_elementt<0,Tup>::type T1;
        typedef typename std::tuple_elementt<1,Tup>::type T2;
        typedef typename std::tuple_elementt<2,Tup>::type T3;
        typedef typename std::tuple_elementt<3,Tup>::type T4;
        typedef typename std::tuple_elementt<4,Tup>::type T5;
        typedef typename std::tuple_elementt<5,Tup>::type T6;
        typedef typename std::tuple_elementt<6,Tup>::type T7;
        typedef typename std::tuple_elementt<7,Tup>::type T8;
        typedef typename std::tuple_elementt<8,Tup>::type T9;
        typedef typename std::tuple_elementt<9,Tup>::type BodyContent;
          
        static void call(const RuntimeConfig& rtcfg, const CSRWrapper& graph, BackwardGData<T3, T4>* gdata)
       // static void call(std::function<void(const RuntimeConfig&,const CSRWrapper&,BackwardGData<T3,T4>*)> body)
        {
           if (gdata->lhs_mapping || gdata->rhs_mapping || gdata->out_mapping) {
              // cpu::FallbackCallBackwardBinaryReduce<Tup>::call(rtcfg, graph, gdata);
              cpu::FallbackCallBackwardBinaryReduce<T1,T2,T3,T4,T5,T6,T7,T8>(rtcfg, graph, gdata);
             } else {
               BodyContent::call(rtcfg,graph,gdata);
             }         
        }
};


struct BContent {

static call(const RuntimeConfig& rtcfg,const CSRWrapper& graph,BackwardGData<int32_t, double>* gdata )
{ 
     auto csr = graph.GetOutCSRMatrix();
     cpu::sparse_mm3(rtcfg, csr, gdata->lhs_data, gdata->out_data, gdata->x_length)
}

};

using ListOfType = std::tuple<kDLCPU,binary_op::kGradLhs,int32_t,double,SelectSrc, SelectNone,BinaryUseLhs<double>, ReduceSum<kDLCPU,double> , BContent>; 
SchemaCallBackBinReduce<ListOfType>::call(rtcfg,graph,gdata);



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



