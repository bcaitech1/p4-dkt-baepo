import { React, useState } from "react";
import { Link } from "react-router-dom";
import "./Navigation.css";

const Navigation = () => {
  const [expand, setExpand] = useState(false);
  const menuButton = document.getElementsByClassName("menu__button")[0]; // 햄버거바
  const navBarItems = document.getElementsByClassName(
    "nav__item__container"
  )[0]; // 메뉴 아이템 링크

  // 스크롤바를 조금이라도 내리게 되면 navbar의 bottom border에 색을 입히는 효과가 추가됨.
  window.addEventListener("scroll", (e) => {
    const navbar = document.getElementsByClassName("nav");
    if (window.scrollY) {
      navbar[0].classList.add("nav__bottom__border");
    } else {
      navbar[0].classList.remove("nav__bottom__border");
    }
  });

  // 햄버거바 메뉴버튼 클릭이벤트
  // menuButton[0].addEventListener("click", () => {
  //   menuButton.classList.toggle("active");
  // });

  return (
    <nav className="nav">
      <div className="nav__container">
        <div className="nav__home__container">
          <Link to="/" className="nav__home">
            Baepo
          </Link>
        </div>
        <div className="nav__on__mobile__container">
          <button className="menu__button" onClick={() => setExpand(!expand)}>
            {expand ? (
              <i className="fas fa-times" />
            ) : (
              <i className="fas fa-bars" />
            )}
          </button>
        </div>
        <div className="nav__item__container">
          <div className="nav__item">
            <Link to="/task" className="nav__task">
              About DKT
            </Link>
          </div>
          <div className="nav__item">
            <Link to="/eda" className="nav__eda">
              Data EDA
            </Link>
          </div>
          <div className="nav__item">
            <Link to="/model" className="nav__model">
              Model Analysis
            </Link>
          </div>
          <div className="nav__item">
            <a
              href="https://boostcamp.connect.or.kr/program_ai.html"
              className="nav__bcaitech"
              target="blank"
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
