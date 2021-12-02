import React from "react";
import { Link } from "react-router-dom";
import "./Navigation.css";

const Navigation = () => {
  return (
    <nav className="nav">
      <div className="nav__container">
        <div className="nav__container__inner">
          <div className="nav__home__container">
            <Link to="/" className="nav__home">
              Baepo
            </Link>
          </div>
          <div className="nav__on__mobile__container"></div>
          <div className="nav__item__container">
            <div className="nav__item">
              <Link to="/task" className="nav__task">
                About_DKT
              </Link>
            </div>
            <div className="nav__item">
              <Link to="/eda" className="nav__eda">
                Data_EDA
              </Link>
            </div>
            <div className="nav__item">
              <Link to="/model" className="nav__model">
                Model_Analysis
              </Link>
            </div>
            <div className="nav__item">
              <a
                href="https://boostcamp.connect.or.kr/program_ai.html"
                className="nav__bcaitech"
                target="blank"
              >
                boostcamp
              </a>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
