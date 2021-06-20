import React from "react";

// 모델 score와 두 개의 플롯을 띄운다.
const Analysis = (accuracy, auroc, lgbm_plot, zero_one) => {
  console.log(accuracy);
  console.log(auroc);
  console.log(lgbm_plot);
  console.log(zero_one);
  return <span>This is Analysis!</span>;
};

export default Analysis;
