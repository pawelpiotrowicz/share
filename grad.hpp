

void BuildMatMulGradDynamic(
    const std::shared_ptr<paddle::framework::OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map)
        {
   std::cout << "NGRAPH_CUSTOM_MATMUL_BACKWARD" << std::endl;
  auto op_attrs = paddle::framework::AttrReader(op->Attrs());
  std::cout << "***** BuildMatMulGradNodeDynamic ***** " << std::endl;
  auto dout = paddle::platform::GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto y = paddle::platform::GetInputNode(op, "Y", ngb_node_map);
  auto x = paddle::platform::GetInputNode(op, "X", ngb_node_map);

  bool is_dx = paddle::platform::HasOutput(op, "X@GRAD") ? true : false;
  bool is_dy = paddle::platform::HasOutput(op, "Y@GRAD") ? true : false;
  bool transpose_x = op_attrs.Get<bool>("transpose_X");
  bool transpose_y = op_attrs.Get<bool>("transpose_Y");
  float alpha = op_attrs.Get<float>("alpha");


 

  // check if fused_op[0] ... fused_op[1]
  
 auto fused_op = std::make_shared<ngraph::op::MatMulPdBackward>( x, y, dout, is_dx, is_dy, transpose_x, transpose_y);
 

  if (is_dx)
  {
    std::shared_ptr<ngraph::Node> dx_t = std::make_shared<ngraph::op::GetOutputElement>(fused_op, 0);
    auto dx_scale = ElementwiseScalar<ngraph::op::Multiply>(1 / alpha, dx_t);
    paddle::platform::SetOutputNode(op, "X@GRAD", fused_op, ngb_node_map);
  }  
  if (is_dy)
  {   
    std::shared_ptr<ngraph::Node> dy_t = std::make_shared<ngraph::op::GetOutputElement>(fused_op, 1); 
    auto dy_scale = ElementwiseScalar<ngraph::op::Multiply>(1 / alpha, dy_t);
    paddle::platform::SetOutputNode(op, "Y@GRAD", fused_op, ngb_node_map);
  }
 }

// ################################

  constexpr NodeTypeInfo op::MatMulPdBackward::type_info;
  op::MatMulPdBackward::MatMulPdBackward(
      std::shared_ptr<ngraph::Node> A,
      std::shared_ptr<ngraph::Node> B,
      std::shared_ptr<ngraph::Node> OutGrad,
      bool is_X, 
      bool is_Y,
      bool transpose_a, 
      bool transpose_b ) : FusedOp(OutputVector{A, B, OutGrad}),
      m_A{A}, m_B{B}, m_is_X{is_X}, m_is_Y(is_Y),
      m_transpose_a{transpose_a}, m_transpose_b{transpose_b}
      {
         constructor_validate_and_infer_types();
      }

  ............
}
   NodeVector op::MatMulPdBackward::decompose_op() const 
      {
            auto x = input_value(0).get_node_shared_ptr();
            auto y = input_value(1).get_node_shared_ptr();
            auto dout = input_value(2).get_node_shared_ptr();
     
        //  auto& dout = OutGrad;
        //  auto& x = m_A;
        //  auto& y = m_B;
         auto dout_shape = dout->get_shape();
         auto x_shape = x->get_shape();
         auto y_shape = y->get_shape();
         size_t nx = x_shape.size();
         size_t ny = y_shape.size();
         size_t ndout = dout_shape.size();
         std::shared_ptr<ngraph::Node> x2, y2;
         std::shared_ptr<ngraph::Node> dout2;

  x2 = helper_transposeAndFlat3D(x, false);
  y2 = helper_transposeAndFlat3D(y, false, false);
  dout2 = helper_transposeAndFlat3D(dout, false);
  auto x2_shape = x2->get_shape();
  auto y2_shape = y2->get_shape();
  if (nx >= 3 || ny >= 3) {
    std::shared_ptr<ngraph::Node> dout_temp;
    if (ndout == 2) {
      dout_temp = std::make_shared<ngraph::op::Reshape>(
          dout, ngraph::AxisVector{0, 1},
          ngraph::Shape{dout_shape[0], dout_shape[1], 1});
      if (ny < 3) {
        dout2 = dout_temp;
      } else {
        dout2 = helper_transposeAndFlat3D(dout_temp, true);
      }
    }
    x2 = helper_broadcast3D(x2, y_shape[0]);
    y2 = helper_broadcast3D(y2, x_shape[0]);

  } else {
    dout2 = helper_transposeAndFlat3D(dout, false, nx == 1 && m_transpose_a == false);
  }

  if (m_transpose_b == false) {
    y2 = helper_transposeAndFlat3D(y2, true);
  }
  if (m_transpose_a == false) {
    x2 = helper_transposeAndFlat3D(x2, true);
  }
  auto dx = helper_dotOp(dout2, y2);
  auto dy = helper_dotOp(x2, dout2);
  if (m_transpose_a == true) {
    dx = helper_transposeAndFlat3D(dx, true);
  }
  if (m_transpose_b == true) {
    dy = helper_transposeAndFlat3D(dy, true);
  }

  if (nx < 3 && ny >= 3) {
    dx = std::make_shared<ngraph::op::Sum>(dx, ngraph::AxisSet{0});
  }
  if (ny < 3 && nx >= 3) {
    dy = std::make_shared<ngraph::op::Sum>(dy, ngraph::AxisSet{0});
  }
  
    auto dx_t = helper_reshapeToOriginal(dx, x_shape);
    auto dy_t = helper_reshapeToOriginal(dy, y_shape);

    return NodeVector{ dx_t, dy_t, dout };

}





