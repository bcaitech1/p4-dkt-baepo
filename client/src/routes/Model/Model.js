import React from "react";
import axios from "axios";
import * as Papa from "papaparse";
import "./Model.css";

class Model extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      inputFile: undefined,
      infResult: undefined,
      isLoading: false,
      // converted: "",
    };
    this.convertCSVtoString = this.convertCSVtoString.bind(this);
    this.getModelInference = this.getModelInference.bind(this);
  }

  // csv를 string으로 파싱하고 converted state를 업데이트한다.
  convertCSVtoString() {
    const { inputFile } = this.state;
    // console.log(inputFile);
    if (inputFile === undefined) {
      alert("You forgot data!🤭");
    } else {
      // csv를 파싱하고 객체에서 데이터만 받아와서 string화 한다.
      Papa.parse(inputFile, {
        complete: (results) => {
          this.setState({ converted: results.data.slice(1).toString() });
        },
      });
    }
  }

  getModelInference() {
    // 로딩화면으로 전환하기 위해 isLoading state를 true 한다.
    this.setState({ isLoading: true });

    // model inference하고 결과 받아오기
    const { inputFile } = this.state;
    if (inputFile === undefined) {
      // input없이 화면이 넘어오면 alert
      alert("You forgot data!🤭");
    } else {
      // 서버 통신
      setTimeout(() => {
        console.log("3초 뒤에 서버에서 값 받음!!");
        this.setState({ infResult: 0.8498 });
      }, 3000);
      // csv를 파싱하고 객체에서 데이터만 받아와서 string화 한다.
      // Papa.parse(inputFile, {
      //   complete: (results) => {
      //     this.setState({ converted: results.data.slice(1).toString() });
      //   },
      // });
    }
  }

  render() {
    const { isLoading, infResult } = this.state;
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
          <button onClick={this.getModelInference}>Inference🔎</button>
        </div>
      );
    } else {
      // isLoading이 true인데 infResult가 undefined이면 로딩화면에서 대기한다.
      if (infResult === undefined) {
        return <div>Loading...</div>;
      } else {
        return <div>score!!</div>;
      }
    }
  }
}

export default Model;
