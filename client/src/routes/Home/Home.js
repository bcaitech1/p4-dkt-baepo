import React from "react";
import { Link } from "react-router-dom";
import "./Home.css";

function Home() {
  return (
    <div className="container">
      <ul className="menu__items">
        <li className="menu__item">
          <Link to="/crews" className="menu__crews">
            <em className="category">Introduction</em>
            <div className="title">Crew Intro</div>
            <div className="sub_info">@Baepo crews</div>
          </Link>
        </li>
        <li className="menu__item bg_blue">
          <div className="menu__about">
            <div className="title ft_white">About us</div>
            <div className="box_info">
              <em className="category ft_white">Baepo는?</em>
              <p className="hover_text">
                boostcamp AI Tech 1기 캠퍼들이 만든 DKT task 체험 공간입니다.
              </p>
            </div>
          </div>
        </li>
        <li className="menu__item">
          <Link to="/task" className="menu__task">
            <em className="category">DKT</em>
            <div className="title">Task Overview</div>
            <div className="sub_info">@someone</div>
          </Link>
        </li>
        <li className="menu__item">
          <Link to="/modeling" className="menu__modeling">
            <em className="category">Dev story</em>
            <div className="title">Modeling</div>
            <div className="sub_info">@someone</div>
          </Link>
        </li>
        <li className="menu__item">
          <Link to="/frontend" className="menu__frontend">
            <em className="category">Dev story</em>
            <div className="title">Frontend</div>
            <div className="sub_info">@sunhwan</div>
          </Link>
        </li>
        <li className="menu__item">
          <Link to="/backend" className="menu__backend">
            <em className="category">Dev story</em>
            <div className="title">Backend</div>
            <div className="sub_info">@someone</div>
          </Link>
        </li>
      </ul>
    </div>
  );
}

export default Home;
