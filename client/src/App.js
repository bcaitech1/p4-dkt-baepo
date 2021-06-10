import React from "react";
import { BrowserRouter, Redirect, Route, Switch } from "react-router-dom";
import "./App.css";
import Navigation from "./components/Navigation/Navigation";
import Home from "./routes/Home/Home";
import Crews from "./routes/Crews/Crews";
import Task from "./routes/Task/Task";
import Eda from "./routes/Eda/Eda";
import Model from "./routes/Model/Model";
import Analysis from "./routes/Analysis/Analysis";

const App = () => {
  return (
    <BrowserRouter>
      <Navigation />
      <Switch>
        <Route path="/" exact component={Home} />
        <Route path="/crews" component={Crews} />
        <Route path="/task" component={Task} />
        <Route path="/eda" component={Eda} />
        <Route path="/model" component={Model} />
        <Route path="/analysis" component={Analysis} />
        <Redirect path="*" to="/" />
      </Switch>
    </BrowserRouter>
  );
};

export default App;
