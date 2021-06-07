import React from "react";
import { Link } from "react-router-dom";
import { BsArrowUpRight } from "react-icons/bs";
import "./Navigation.css";

function Navigation() {
  return (
    <nav className="nav">
      <Link to="/" className="nav__home">
        Baepo
      </Link>
      <div className="nav__items">
        <Link to="/about" className="nav__about">
          About
          <BsArrowUpRight className="nav__arrow"></BsArrowUpRight>
        </Link>
        <Link to="/crews" className="nav__crews">
          Crews
          <BsArrowUpRight className="nav__arrow"></BsArrowUpRight>
        </Link>
        <a
          href="https://boostcamp.connect.or.kr/program_ai.html"
          class="nav__bcaitech"
          target="blank"
        >
          boostcamp
          <BsArrowUpRight className="nav__arrow"></BsArrowUpRight>
        </a>
      </div>
    </nav>
  );
}

export default Navigation;
