import { React, useState } from "react";
import { Link } from "react-router-dom";
import "./Navigation.css";

const Navigation = () => {
  const [expand, setExpand] = useState(false); // 모바일뷰 메뉴 상태(expanded, not expanded)

  // 스크롤바를 조금이라도 내리게 되면 navbar의 bottom border에 색을 입히는 효과가 추가됨.
  window.addEventListener("scroll", (e) => {
    const navbar = document.getElementsByClassName("nav");
    if (window.scrollY) {
      navbar[0].classList.add("nav__bottom__border");
    } else {
      navbar[0].classList.remove("nav__bottom__border");
    }
  });

  console.log(expand);

  return (
    <nav className="nav">
      <div className="nav__container">
        <div className="nav__home__container">
          <Link
            to="/"
            className="nav__home"
            onClick={() => {
              setExpand(false);
            }}
          >
            Baepo
          </Link>
        </div>
        <div className="nav__on__mobile__container">
          <button className="menu__button" onClick={() => setExpand(!expand)}>
            {expand ? (
              <i className="close-button fas fa-times" />
            ) : (
              <i className="open-button fas fa-bars" />
            )}
          </button>
        </div>
        <div
          className={
            expand ? "nav__item__container active" : "nav__item__container"
          }
        >
          <div className="nav__item">
            <Link
              to="/task"
              className="nav__task"
              onClick={() => {
                setExpand(false);
              }}
            >
              About DKT
            </Link>
          </div>
          <div className="nav__item">
            <Link
              to="/eda"
              className="nav__eda"
              onClick={() => {
                setExpand(false);
              }}
            >
              Data EDA
            </Link>
          </div>
          <div className="nav__item">
            <Link
              to="/model"
              className="nav__model"
              onClick={() => {
                setExpand(false);
              }}
            >
              Model Analysis
            </Link>
          </div>
          <div className="nav__item">
            <a
              href="https://boostcamp.connect.or.kr/program_ai.html"
              className="nav__bcaitech"
              target="blank"
              onClick={() => {
                setExpand(false);
              }}
            >
              boostcamp AI Tech
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
