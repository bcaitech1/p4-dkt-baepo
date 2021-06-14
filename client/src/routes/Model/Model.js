import React from "react";
import axios from "axios";
import * as Papa from "papaparse";
import "./Model.css";

class Model extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      startInf: false,
      inputFile: undefined,
      infResult: undefined,
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
    // model inference하고 결과 받아오기
    this.setState({ infResult: true });
  }

  render() {
    const { startInf, infResult } = this.state;
    console.log("rendering...."); // render function이 call된 것을 확인

    if (startInf === false) {
      // inference button
      return (
        <button
          onClick={() => {
            this.setState({ startInf: true });
          }}
        >
          Start Inference🧑🏻‍💻
        </button>
      );
    } else {
      // start inference
      return <span>{this.state.converted}</span>;
    }
  }
}

export default Model;

// state가 변경되면 rendering이 다시 일어난다.
// csv file input을 받아서 string으로 변환하고,
// 만약, 변환을 마치고 converted state가 선언이 되면 서버 업로드 컴포넌트를 보게 된다.
// if (infResult === undefined) {
//   return (
//     <div className="file_upload">
//       <label>Test Data Input(CSV file)</label>
//       <input
//         type="file"
//         accept=".csv"
//         onChange={(event) => {
//           this.setState({ inputFile: event.target.files[0] });
//         }}
//       />
//       <button onClick={this.getModelInference}>Inference🔎</button>
//     </div>
//   );
// } else {
//   return <div>server component</div>;
// }
