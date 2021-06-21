import React from "react";
import axios from "axios";
import dotenv from "dotenv";
// import * as Papa from "papaparse";
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
      // converted: "",
    };
    // this.convertCSVtoString = this.convertCSVtoString.bind(this);
    this.modelInference = this.modelInference.bind(this);
    this.getModelScore = this.getModelScore.bind(this);
  }

  getModelScore = async () => {
    const { inputFile } = this.state;
    let formData = new FormData();
    formData.append("data", inputFile);
    const score = await axios.post(
      process.env.REACT_APP_SERVER + "/inference",
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    this.setState({ infScore: score });
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
            Test Data Input(CSV file)
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
          <button onClick={this.modelInference}>Inference🔎</button>
        </div>
      );
    } else {
      // inference 결과를 받기 전까지 로딩화면을 띄운다.
      if (infScore === undefined) {
        return <div>Loading...</div>;
      } else {
        // inference 결과로 받아온 모델 score와 플롯 두개를 props로 넘겨주고,
        // analysis 컴포넌트에서 보여준다.
        console.log(this.state.infScore);
        const {
          prediction,
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

// csv를 string으로 파싱하고 converted state를 업데이트한다.
// convertCSVtoString() {
//   const { inputFile } = this.state;
//   // console.log(inputFile);
//   if (inputFile === undefined) {
//     alert("You forgot data!🤭");
//   } else {
//     // csv를 파싱하고 객체에서 데이터만 받아와서 string화 한다.
//     Papa.parse(inputFile, {
//       complete: (results) => {
//         this.setState({ converted: results.data.slice(1).toString() });
//       },
//     });
//   }
// }
