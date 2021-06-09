// 메뉴의 항목을 구성하는 컴포넌트입니다.
import React from "react";
import { Link } from "react-router-dom";

function MenuItem({ name, category, title, sub_info }) {
  return (
    <Link to={`/${name}`} className={`menu__${name}`}>
      <em className="category">{category}</em>
      <div className="title">{title}</div>
      <div className="sub_info">{sub_info}</div>
    </Link>
  );
}

export default MenuItem;
