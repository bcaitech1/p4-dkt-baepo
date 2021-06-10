import React from "react";
import "./Overview.css";

function Overview() {
  return (
    <div className="task__content">
      <p>
        초등학교, 중학교, 고등학교, 대학교와 같은 교육기관에서 우리는 시험을 늘
        봐왔습니다. 시험 성적이 높은 과목은 우리가 잘 아는 것을 나타내고 시험
        성적이 낮은 과목은 반대로 공부가 더욱 필요함을 나타냅니다. 시험은 우리가
        얼마만큼 아는지 평가하는 한 방법입니다.
      </p>
      <p>
        하지만 시험에는 한계가 있습니다. 우리가 수학 시험에서 점수를 80점
        받았다면 우리는 80점을 받은 학생일 뿐입니다. 우리가 돈을 들여 과외를
        받지 않는 이상 우리는 우리 개개인에 맞춤화된 피드백을 받기가 어렵고
        따라서 무엇을 해야 성적을 올릴 수 있을지 판단하기 어렵습니다. 이럴 때
        사용할 수 있는 것이 DKT입니다!
      </p>
      <p>
        DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는
        딥러닝 방법론입니다.
      </p>
      <p>
        <br />
      </p>
      <img
        src="https://s3-ap-northeast-2.amazonaws.com/prod-aistages-public/app/Users/00000068/files/f74f6ce6-dfd8-47fa-8a57-cff8c2f89c3f..png"
        contenteditable="false"
        draggable="true"
        alt="task_image"
      />
      <br />
      <p>
        <br />
      </p>
      <p>
        시험을 보는 것은 동일하지만 단순히 우리가 수학을 80점을 맞았다고
        알려주는 것을 넘어서 우리가 수학이라는 과목을 얼마만큼 이해하고 있는지를
        측정하여 줍니다. 게다가 이런 이해도를 활용하여 우리가 아직 풀지 않은
        미래의 문제에 대해서 우리가 맞을지 틀릴지 예측이 가능합니다!
      </p>
      <p>
        <br />
      </p>
      <img
        src="https://s3-ap-northeast-2.amazonaws.com/prod-aistages-public/app/Users/00000068/files/72bf5780-c031-4d31-8a01-06e4e7403454..png"
        contenteditable="false"
        draggable="true"
        alt="task_image"
      />
      <br />
      <p>
        <br />
      </p>
      <p>
        이런 DKT를 활용하면 우리는 학생 개개인에게 수학의 이해도와 취약한 부분을
        극복하기 위해 어떤 문제들을 풀면 좋을지 추천이 가능합니다! 그렇기 때문에
        DKT는 교육 AI의 추천이라고도 불립니다. DKT는 맞춤화된 교육을 제공하기
        위해 아주 중요한 역할을 맡게 됩니다.
      </p>
      <p>
        <br />
      </p>
      <img
        src="https://s3-ap-northeast-2.amazonaws.com/prod-aistages-public/app/Users/00000068/files/57a525f9-a799-49a5-84f6-3150c8fb6ccb..png"
        contenteditable="false"
        draggable="true"
        alt="task_image"
      />
      <br />
      <p>
        <br />
      </p>
      <p>
        우리는 대회에서 Iscream 데이터셋을 이용하여 DKT모델을 구축할 것입니다.
        다만 대회에서는 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는
        일보다는, 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중할 것입니다!
        우리는 각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 최종
        문제를 맞출지 틀릴지 예측할 것입니다!
      </p>
      <p>
        <br />
      </p>
      <img
        src="https://s3-ap-northeast-2.amazonaws.com/prod-aistages-public/app/Users/00000068/files/1d284236-be6f-44aa-a4bd-fb4c5f8c6a12..png"
        contenteditable="false"
        draggable="true"
        alt="task_image"
      />
      <br />
      <p>
        <br />
      </p>
      <p>
        개인 맞춤화 교육이라는 멋진 미래를 만들기 위한 DKT로 함께 떠나봅시다!
      </p>
    </div>
  );
}

export default Overview;
