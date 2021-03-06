import React from "react";
import axios from "axios";
import dotenv from "dotenv";
import Spinner from "react-bootstrap/Spinner";

import "./Model.css";
import { Analysis } from "../../pages";

dotenv.config();

class Model extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      inputFile: undefined,
      infScore: undefined,
      isLoading: false,
    };
    this.getModelScore = this.getModelScore.bind(this);
    this.modelInference = this.modelInference.bind(this);
  }

  getModelScore = async () => {
    const { inputFile } = this.state;
    let formData = new FormData();
    formData.append("data", inputFile);
    try {
      const response = await axios.post(
        process.env.REACT_APP_SERVER + "/inference",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      this.setState({ infScore: response.data });
    } catch (err) {
      console.log("Error 발생!!!: " + err);
    }
  };

  modelInference() {
    // model inference하고 결과 받아오기
    const { inputFile } = this.state;
    if (inputFile === undefined) {
      alert("You forgot data!🤭"); // input없이 화면이 넘어오면 alert
    } else {
      this.setState({ isLoading: true }); // isLoading state를 true하며 로딩화면으로 렌더링
      this.getModelScore(); // 모델 서버와 통신(서버 내에 raw 데이터 생성, 플롯을 response로 받는다.)
    }
  }

  render() {
    const { isLoading, infScore } = this.state;
    console.log("rendering...."); // render function이 call된 것을 확인
    // state가 변경되면 rendering이 다시 일어난다.
    // csv file input을 받으면 button onclick 콜백함수로 modelInference가 실행되고
    // isLoading state가 true가 되며 로딩화면이 뜬다.
    if (isLoading === false) {
      return (
        <div className="file_upload">
          <label className="file_label" htmlFor="file">
            📂 Input file(csv) 📂
          </label>
          <input
            id="file"
            className="file_input"
            type="file"
            accept=".csv"
            onChange={(event) => {
              this.setState({
                inputFile: event.target.files[0],
              });
            }}
          />
          <button onClick={this.modelInference}>Start Inference🔎</button>
        </div>
      );
    } else {
      // modelInference 메서드에서 inputFile이 들어온 것을 체크해서 isLoading이 true가 된다면
      // inference 결과를 받기 전까지 로딩화면을 띄운다.
      if (infScore === undefined) {
        return (
          <div className="loading__container">
            <Spinner
              className="loading__logo"
              animation="border"
              variant="primary"
            />
          </div>
        );
      } else {
        // inference 결과로 받아온 모델 score와 플롯 두개를 props로 넘겨주고,
        // analysis 컴포넌트에서 보여준다.
        const {
          accuracy_score,
          roc_auc_score,
          lgbm_plot_importance,
          zero_one_distribution,
        } = this.state.infScore;
        return (
          <Analysis
            accuracy={accuracy_score}
            auroc={roc_auc_score}
            lgbm_plot={lgbm_plot_importance}
            zero_one={zero_one_distribution}
          />
        );
      }
    }
  }
}

export default Model;
