import React from "react";
import { BrowserRouter, Redirect, Route, Switch } from "react-router-dom";
import "./App.css";
import Navigation from "./components/Navigation/Navigation";
import Home from "./routes/Home/Home";
import Crews from "./routes/Crews/Crews";
import Task from "./routes/Task/Task";

function App() {
  return (
    <BrowserRouter>
      <Navigation />
      <Switch>
        <Route path="/" exact component={Home} />
        <Route path="/crews" component={Crews} />
        <Route path="/task" component={Task} />
        <Redirect path="*" to="/" />
      </Switch>
    </BrowserRouter>
  );
}

export default App;
