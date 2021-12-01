import React from "react";
import "./Analysis.css";

// 모델 score와 두 개의 플롯을 띄운다.
const Analysis = ({ accuracy, auroc, lgbm_plot, zero_one }) => {
  console.log("1: " + accuracy);
  console.log("2: " + auroc);
  console.log("3: " + lgbm_plot);
  console.log("4: " + zero_one);
  return (
    <div className="analysis__container">
      <div className="analysis__scores">
        <span className="analysis__score">Accuracy: {accuracy.toFixed(4)}</span>
        <span className="analysis__score">AUROC: {auroc.toFixed(4)}</span>
      </div>
      <div className="analysis__plots">
        <img
          className="lgbm_plot"
          src={`${process.env.REACT_APP_SERVER}/static/${lgbm_plot}`}
          alt="lgbm_plot"
        />
        <img
          className="zero_one"
          src={`${process.env.REACT_APP_SERVER}/static/${zero_one}`}
          alt="zero_one"
        />
        <div>
          학생들의 풀이가 전반적으로 1을 더 잘 맞추는 쪽으로 분포가 잘 형성된
          것을 볼 수 있다.
        </div>
      </div>
    </div>
  );
};

export default Analysis;
