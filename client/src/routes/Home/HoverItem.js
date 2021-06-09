// mouse hover 이벤트가 있는 메뉴 컴포넌트입니다.
// hover 이벤트 컴포넌트가 또 필요할 경우 좀더 재사용성을 고려하여 리팩토링하면 될 것 같습니다.
import React from "react";

function HoverItem() {
  return (
    <div className="menu__about">
      <div className="title ft_white">About us</div>
      <div className="box_info">
        <em className="category ft_white ft_bold">Team Baepo는?</em>
        <p className="hover_text">
          저희는 boostcamp AI Tech 1기
          <br />
          Team Baepo 입니다🔥
          <br />
          P stage 4 DKT task를 진행하면서 서비스화 한다는
          <br />
          마음가짐으로 프로젝트를 진행했습니다🌱
          <br />
          프로젝트의 상세한 내용은 파트별로 나누어 <br />
          담아보았습니다🙌🏼
        </p>
      </div>
    </div>
  );
}

export default HoverItem;
