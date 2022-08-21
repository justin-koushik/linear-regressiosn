const can=document.querySelector('canvas')
const table=document.querySelector('table')
const slope=document.querySelector('.slope')
const inter=document.querySelector('.inter')
const error=document.querySelector('.error')
const posX=document.querySelector('.posx')
const posY=document.querySelector('.posy')
const w=500
const h=500
can.width=w
can.height=h
can.style.border='1px solid'
const ctx=can.getContext('2d')

let actX=0
let actY=0
let filters=[]

let m=tf.variable(tf.scalar(Math.random()))
let c=tf.variable(tf.scalar(Math.random()))
slope.textContent=m.dataSync()[0]
inter.textContent=c.dataSync()[0]
let x=[]
let y=[]
function map(x,l1,h1,l2,h2){
    return ((x-l1)/(h1-l1))*(h2-l2)+l2
}
const lossfunc=(pred,real)=>tf.losses.meanSquaredError(pred,real)
const optimizer=tf.train.sgd(0.1)
const model=(X)=>X.mul(m).add(c)
ctx.fillStyle="red"
let x1=tf.tensor(-1)
let x2=tf.tensor(1)
function draw(){
    ctx.strokeStyle='rgba(0,0,0,0.5)'
    ctx.lineWidth=0.4
    for(let i=0;i<w;i=i+10){
        ctx.beginPath()
        ctx.moveTo(i,0)
        ctx.lineTo(i,h)
        ctx.stroke()
        ctx.closePath()
    }
    for(let i=0;i<h;i=i+10){
        ctx.beginPath()
        ctx.moveTo(0,i)
        ctx.lineTo(w,i)
        ctx.stroke()
        ctx.closePath()
    }
    ctx.lineWidth=1
    ctx.strokeStyle='black'
}
function animate(){
    requestAnimationFrame(animate)
    const preds=document.querySelectorAll('.pred')
    ctx.clearRect(0,0,w,h)
    draw()
    if(x.length>0){
        tf.tidy(()=>{
            optimizer.minimize(()=>lossfunc(model(tf.tensor(x)),tf.tensor(y)))
            slope.textContent=(m.dataSync()[0]).toFixed(3)
            inter.textContent=(c.dataSync()[0]).toFixed(3)
            error.textContent=(lossfunc(tf.tensor(x),tf.tensor(y)).dataSync()[0]).toFixed(3)
            let y1=model(x1)
            let y2=model(x2)
            ctx.beginPath()
            ctx.moveTo(0,map(y1.dataSync()[0],-1,1,0,h))
            ctx.lineTo(w,map(y2.dataSync()[0],-1,1,0,h))
            ctx.stroke()
            ctx.closePath()
            preds.forEach((e,idx)=>{
                e.textContent=Math.round(map(model(tf.tensor(x[idx])).dataSync()[0],-1,1,0,h))
            })
        })
    }
    for(let i=0;i<x.length;i++){
        ctx.beginPath()
        ctx.arc(map(x[i],-1,1,0,w),map(y[i],-1,1,0,h),2,0,Math.PI*2)
        ctx.fill()
        ctx.closePath()
    }
    for(let point of filters){
        ctx.beginPath()
        ctx.rect(point[0]-3,point[1]-3,6,6)
        ctx.stroke()
        ctx.closePath()
    }
}
animate()
can.onclick=(e)=>{
    filters=[]
    x.push(map(e.x,0,w,-1,1))
    y.push(map(e.y,0,h,-1,1))
    let tr=document.createElement('tr')
    let td1=document.createElement('td')
    let td2=document.createElement('td')
    let td3=document.createElement('td')
    td3.classList.add('pred')
    tf.tidy(()=>{
        let xx=tf.tensor(map(e.x,0,w,-1,1))
        let yy=model(xx)
        td3.textContent=map(yy.dataSync()[0],-1,1,0,h)
        
    })
    td1.textContent=e.x
    td2.textContent=e.y
    tr.appendChild(td1)
    tr.appendChild(td2)
    tr.appendChild(td3)
    td1.setAttribute('id','x')
    td2.setAttribute('id','y')
    table.appendChild(tr)
}
can.onmousemove=(e)=>{
    posX.textContent=e.x
    posY.textContent=e.y
}
table.onclick=(e)=>{
    let id=e.target.id
    let item=e.target.innerText
    if((actX && id==='x') || (actY && id==='y')){
        filters=[]
        actX=0
        actY=0
    }
    if((actX^actY)===0){
        if(id==='x'){
            for(let i=0;i<x.length;i++){
                let xx=map(x[i],-1,1,0,w)
                if(Math.abs(xx-item)<1e-3){
                    filters.push([xx,map(y[i],-1,1,0,h)])
                }
            }
            actX=1
        }else{
            for(let i=0;i<y.length;i++){
                let yy=map(y[i],-1,1,0,h)
                if(Math.abs(yy-item)<1e-3){
                    filters.push([map(x[i],-1,1,0,w),yy])
                }
            }
            actY=1
        }
    }else{
        if(actX && id==='y'){
            filters=filters.filter(e=>{
                return Math.abs(e[1]-item)<1e-3
            })
            actY=1
        }else if(actY && id==='x'){
            filters=filters.filter(e=>{
                return Math.abs(e[0]-item)<1e-3
            })
            actX=1
        }
    }
}
