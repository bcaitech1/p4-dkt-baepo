import React from "react";
import { BrowserRouter, Redirect, Route, Switch } from "react-router-dom";
import "./App.css";
import Navigation from "./components/Navigation";
import { Home, Task, Eda, Model } from "./pages";

const App = () => {
  return (
    <BrowserRouter>
      <Navigation />
      <div className="container">
        <Switch>
          <Route path="/" exact component={Home} />
          <Route path="/task" component={Task} />
          <Route path="/eda" component={Eda} />
          <Route path="/model" component={Model} />
          {/* Link에서 라우터를 찾을 때, 매칭되는 것이 없다면 Home으로 돌아오게 설정함. 이후에 여유가 되면 page not found도 구현하면 될 듯! */}
          <Redirect path="*" to="/" />
        </Switch>
      </div>
    </BrowserRouter>
  );
};

export default App;
