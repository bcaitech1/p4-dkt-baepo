import React from "react";
import "./Home.css";
import MenuItem from "./MenuItem";
import HoverItem from "./HoverItem";
import MenuLists from "./MenuLists";

const Home = () => {
  // 메뉴 객체 리스트를 불러와서 컴포넌트화하고, 배열에 저장한다.
  let MenuComponents = MenuLists.map((menu, index) => (
    <li className="menu__item" key={index}>
      <MenuItem
        name={menu.name}
        category={menu.category}
        title={menu.title}
        sub_info={menu.sub_info}
      />
    </li>
  ));
  // 두 번째에 hover item이 들어갈 예정이므로 추가해준다.
  MenuComponents.splice(
    1,
    0,
    <li className="menu__item bg_blue" key={-1}>
      <HoverItem />
    </li>
  );

  // return <ul className="menu__items">{MenuComponents}</ul>;
  return <div>Temporary Home</div>;
};

export default Home;
