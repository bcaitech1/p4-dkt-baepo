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
    // const score = await axios.post(process.env.REACT_APP_SERVER + "/analysis", {
    //   params: {
    //     data: inputFile,
    //     prob_count: "0",
    //     user_acc: "0.6",
    //     check: "upper",
    //   },
    // });
    // this.setState({ infScore: score });
    setTimeout(() => {
      this.setState({ infScore: 1 });
    }, 3000);
  };

  modelInference() {
    this.setState({ isLoading: true }); // 로딩화면으로 전환하기 위해 isLoading state를 true 한다.

    // model inference하고 결과 받아오기
    const { inputFile } = this.state;
    if (inputFile === undefined) {
      alert("You forgot data!🤭"); // input없이 화면이 넘어오면 alert
    } else {
      this.getModelScore(); // 서버 통신
    }
  }

  render() {
    const { isLoading, infScore } = this.state;
    console.log("rendering...."); // render function이 call된 것을 확인

    // state가 변경되면 rendering이 다시 일어난다.
    // csv file input을 받으면 button onclick 콜백함수에서 getModelInference가 실행되고
    // isLoading state가 true가 되며 로딩화면이 뜬다.
    if (isLoading === false) {
      return (
        <div className="file_upload">
          <label>Test Data Input(CSV file)</label>
          <input
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
        return <Analysis />;
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
