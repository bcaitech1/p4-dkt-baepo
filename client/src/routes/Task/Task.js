import React from "react";
import { Link } from "react-router-dom";

function Task() {
  return (
    <div className="task__container">
      <Link to="/crews">To crews</Link>
    </div>
  );
}
export default Task;
