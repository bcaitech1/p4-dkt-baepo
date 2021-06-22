(this.webpackJsonpclient=this.webpackJsonpclient||[]).push([[0],{43:function(e,t,c){},44:function(e,t,c){},51:function(e,t,c){},52:function(e,t,c){},53:function(e,t,c){},75:function(e,t,c){},76:function(e,t,c){"use strict";c.r(t);var a=c(0),s=c.n(a),n=c(28),o=c.n(n),i=c(9),r=c(3),l=(c(43),c(12)),j=(c(44),c(1)),b=function(){return Object(j.jsxs)("nav",{className:"nav",children:[Object(j.jsx)(i.b,{to:"/",className:"nav__home",children:"Team Baepo"}),Object(j.jsxs)("div",{className:"nav__items",children:[Object(j.jsxs)(i.b,{to:"/crews",className:"nav__crews",children:["Crews",Object(j.jsx)(l.a,{className:"nav__arrow"})]}),Object(j.jsxs)(i.b,{to:"/model",className:"nav__model",children:["Model",Object(j.jsx)(l.a,{className:"nav__arrow"})]}),Object(j.jsxs)("a",{href:"https://boostcamp.connect.or.kr/program_ai.html",className:"nav__bcaitech",target:"blank",children:["boostcamp",Object(j.jsx)(l.a,{className:"nav__arrow"})]})]})]})},d=(c(51),function(e){var t=e.name,c=e.category,a=e.title,s=e.sub_info;return Object(j.jsxs)(i.b,{to:"/".concat(t),className:"menu__".concat(t),children:[Object(j.jsx)("em",{className:"category",children:c}),Object(j.jsx)("div",{className:"title",children:a}),Object(j.jsx)("div",{className:"sub_info",children:s})]})}),m=function(){return Object(j.jsxs)("div",{className:"menu__about",children:[Object(j.jsx)("div",{className:"title ft_white",children:"About us"}),Object(j.jsxs)("div",{className:"box_info",children:[Object(j.jsx)("em",{className:"category ft_white ft_bold",children:"Team Baepo\ub294?"}),Object(j.jsxs)("p",{className:"hover_text",children:["\uc800\ud76c\ub294 boostcamp AI Tech 1\uae30",Object(j.jsx)("br",{}),"Team Baepo \uc785\ub2c8\ub2e4\ud83d\udd25",Object(j.jsx)("br",{}),"P stage 4 DKT task\ub97c \uc9c4\ud589\ud558\uba74\uc11c \uc11c\ube44\uc2a4\ud654 \ud55c\ub2e4\ub294",Object(j.jsx)("br",{}),"\ub9c8\uc74c\uac00\uc9d0\uc73c\ub85c \ud504\ub85c\uc81d\ud2b8\ub97c \uc9c4\ud589\ud588\uc2b5\ub2c8\ub2e4\ud83c\udf31",Object(j.jsx)("br",{}),"\ud504\ub85c\uc81d\ud2b8\uc758 \uc0c1\uc138\ud55c \ub0b4\uc6a9\uc740 \ud30c\ud2b8\ubcc4\ub85c \ub098\ub204\uc5b4 ",Object(j.jsx)("br",{}),"\ub2f4\uc544\ubcf4\uc558\uc2b5\ub2c8\ub2e4\ud83d\ude4c\ud83c\udffc"]})]})]})},u=[{name:"crews",category:"Introduction",title:"Crew Intro",sub_info:"@Baepo crews"},{name:"task",category:"Overview",title:"Task Overview",sub_info:"@Baepo crews"},{name:"eda",category:"Pre-stage",title:"Data EDA",sub_info:"@someone"},{name:"model",category:"Modeling",title:"Model & Analysis",sub_info:"@someone"}],h=function(){var e=u.map((function(e,t){return Object(j.jsx)("li",{className:"menu__item",children:Object(j.jsx)(d,{name:e.name,category:e.category,title:e.title,sub_info:e.sub_info})},t)}));return e.splice(1,0,Object(j.jsx)("li",{className:"menu__item bg_blue",children:Object(j.jsx)(m,{})},-1)),Object(j.jsx)("ul",{className:"menu__items",children:e})},_=(c(52),function(e){var t=e.accuracy,c=e.auroc,a=e.lgbm_plot,s=e.zero_one;return console.log("1: "+t),console.log("2: "+c),console.log("3: "+a),console.log("4: "+s),Object(j.jsxs)("div",{className:"analysis__container",children:[Object(j.jsxs)("div",{className:"analysis__scores",children:[Object(j.jsxs)("span",{className:"analysis__score",children:["Accuracy: ",t.toFixed(4)]}),Object(j.jsxs)("span",{className:"analysis__score",children:["AUROC: ",c.toFixed(4)]})]}),Object(j.jsxs)("div",{className:"analysis__plots",children:[Object(j.jsx)("img",{className:"lgbm_plot",src:"".concat("http://54.237.126.194:5000","/static/").concat(a),alt:"lgbm_plot"}),Object(j.jsx)("img",{className:"zero_one",src:"".concat("http://54.237.126.194:5000","/static/").concat(s),alt:"zero_one"}),Object(j.jsx)("div",{children:"\ud559\uc0dd\ub4e4\uc758 \ud480\uc774\uac00 \uc804\ubc18\uc801\uc73c\ub85c 1\uc744 \ub354 \uc798 \ub9de\ucd94\ub294 \ucabd\uc73c\ub85c \ubd84\ud3ec\uac00 \uc798 \ud615\uc131\ub41c \uac83\uc744 \ubcfc \uc218 \uc788\ub2e4."})]})]})}),p=(c(53),function(){return Object(j.jsx)("span",{children:"This is Crews!"})}),O=function(){return Object(j.jsx)("span",{children:"This is EDA!"})},x=c(18),f=c.n(x),v=c(30),g=c(31),N=c(32),y=c(11),w=c(37),k=c(36),S=c(33),T=c.n(S),I=c(34),C=c.n(I),F=c(38);c(74),c(75);C.a.config();var M=function(e){Object(w.a)(c,e);var t=Object(k.a)(c);function c(e){var a;return Object(g.a)(this,c),(a=t.call(this,e)).getModelScore=Object(v.a)(f.a.mark((function e(){var t,c,s;return f.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return t=a.state.inputFile,(c=new FormData).append("data",t),e.prev=3,e.next=6,T.a.post("http://54.237.126.194:5000/inference",c,{headers:{"Content-Type":"multipart/form-data"}});case 6:s=e.sent,a.setState({infScore:s.data}),e.next=13;break;case 10:e.prev=10,e.t0=e.catch(3),console.log(e.t0);case 13:case"end":return e.stop()}}),e,null,[[3,10]])}))),a.state={inputFile:void 0,infScore:void 0,isLoading:!1},a.modelInference=a.modelInference.bind(Object(y.a)(a)),a.getModelScore=a.getModelScore.bind(Object(y.a)(a)),a}return Object(N.a)(c,[{key:"modelInference",value:function(){void 0===this.state.inputFile?alert("You forgot data!\ud83e\udd2d"):(this.setState({isLoading:!0}),this.getModelScore())}},{key:"render",value:function(){var e=this,t=this.state,c=t.isLoading,a=t.infScore;if(!1===c)return Object(j.jsxs)("div",{className:"file_upload",children:[Object(j.jsx)("label",{className:"file_label",htmlFor:"file",children:"\ud83d\udcc2Input file here(.csv)\ud83d\udcc2"}),Object(j.jsx)("input",{id:"file",className:"file_input",type:"file",accept:".csv",onChange:function(t){e.setState({inputFile:t.target.files[0]})}}),Object(j.jsx)("button",{onClick:this.modelInference,children:"Start Inference\ud83d\udd0e"})]});if(void 0===a)return Object(j.jsx)("div",{className:"loading__container",children:Object(j.jsx)(F.a,{className:"loading__logo",animation:"border",variant:"primary"})});var s=this.state.infScore,n=(s.prediction,s.accuracy_score),o=s.roc_auc_score,i=s.lgbm_plot_importance,r=s.zero_one_distribution;return Object(j.jsx)(_,{accuracy:n,auroc:o,lgbm_plot:i,zero_one:r})}}]),c}(s.a.Component),A=function(e){return console.log(e),Object(j.jsx)("span",{children:"This is Task!"})},B=function(){return Object(j.jsxs)(i.a,{children:[Object(j.jsx)(b,{}),Object(j.jsx)("div",{className:"container",children:Object(j.jsxs)(r.d,{children:[Object(j.jsx)(r.b,{path:"/",exact:!0,component:h}),Object(j.jsx)(r.b,{path:"/crews",component:p}),Object(j.jsx)(r.b,{path:"/task",component:A}),Object(j.jsx)(r.b,{path:"/eda",component:O}),Object(j.jsx)(r.b,{path:"/model",component:M}),Object(j.jsx)(r.b,{path:"/analysis",component:_}),Object(j.jsx)(r.a,{path:"*",to:"/"})]})})]})};o.a.render(Object(j.jsx)(s.a.StrictMode,{children:Object(j.jsx)(B,{})}),document.getElementById("root"))}},[[76,1,2]]]);
//# sourceMappingURL=main.d0b43710.chunk.js.map