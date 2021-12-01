import React from "react";
import { Link } from "react-router-dom";
import "./Navigation.css";

const Navigation = () => {
  return (
    <nav className="nav">
      <Link to="/" className="nav__home">
        Team Baepo
      </Link>
      <div className="nav__items">
        <Link to="/task" className="nav__task">
          Task_Overview
        </Link>
        <Link to="/eda" className="nav__eda">
          Data_EDA
        </Link>
        <Link to="/model" className="nav__model">
          Model&Analysis
        </Link>
        <a
          href="https://boostcamp.connect.or.kr/program_ai.html"
          className="nav__bcaitech"
          target="blank"
        >
          boostcamp
        </a>
      </div>
    </nav>
  );
};

export default Navigation;
