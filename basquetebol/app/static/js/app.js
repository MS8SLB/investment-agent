// Menu móvel
document.addEventListener('click',e=>{const b=e.target.closest('[data-menu]');if(b)document.querySelector('.side').classList.toggle('open')});
const NAVY='#0B1F3A',ORANGE='#E8600A',GREY='#9aa3b2';
const PALETA=['#0B1F3A','#E8600A','#2a7ab0','#7a8a2b','#8e44ad','#16a085','#c0392b','#7f8c8d'];
const ptNum=(v,c=2)=>v==null?'—':Number(v).toFixed(c).replace('.',',');
const fmtData=s=>{const [y,m,d]=s.split('-');return `${d}/${m}/${y}`};
// Gráfico de linhas simples (evolução). series=[{label,data:[{x,y}]}]
function linha(el,series,unidade,menorMelhor,casas=2){
  if(el._c)el._c.destroy();
  el._c=new Chart(el,{type:'line',data:{datasets:series.map((s,i)=>({label:s.label,
    data:s.data.map(p=>({x:fmtData(p.x),y:p.y})),borderColor:PALETA[i%PALETA.length],backgroundColor:PALETA[i%PALETA.length],tension:.2,pointRadius:4}))},
    options:{responsive:true,maintainAspectRatio:false,parsing:true,
      scales:{x:{type:'category',labels:[...new Set(series.flatMap(s=>s.data.map(p=>fmtData(p.x))))]},
        y:{title:{display:true,text:unidade+(menorMelhor?'  (menor = melhor)':'  (maior = melhor)')},reverse:false}},
      plugins:{legend:{display:series.length>1},tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${ptNum(c.parsed.y,casas)} ${unidade}`}}}}});
}
function barras(el,labels,values,titulo,cor=ORANGE,extra={}){
  if(el._c)el._c.destroy();
  el._c=new Chart(el,{type:'bar',data:{labels,datasets:[{label:titulo,data:values,backgroundColor:cor}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,title:{display:true,text:titulo}}},...extra}});
}
async function getJSON(u){const r=await fetch(u);if(!r.ok)throw new Error(r.status);return r.json()}
