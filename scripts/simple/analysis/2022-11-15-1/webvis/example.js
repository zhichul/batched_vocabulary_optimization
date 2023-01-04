
var checkpoints = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

var datasets = ["train", "valid"]
var gls = ["0.0"]

var dataset = 0;
var checkpoint = 0;
var example = 0;
var gl = 0;

var scale = 7;
var unit = 1 * scale;
var y_offset = 2 * unit;
var max_thickness = 1 * scale;
var max_font = 3 * scale;
var width = 250 * unit;
var height = 30 * unit;
const curve = d3.line().curve(d3.curveNatural);

d3.select("body").append("h1").text("Attention Marginals")
var svg = d3.select("body").append("svg").attr("width", width).attr("height", height).style("border", "1px dashed");
d3.select("body").append("br")
d3.select("body").append("h1").text("LM Probability")
var svg1 = d3.select("body").append("svg").attr("width", width).attr("height", height).style("border", "1px dashed");
d3.select("body").append("br")
d3.select("body").append("h1").text("LM Marginals")
var svg2 = d3.select("body").append("svg").attr("width", width).attr("height", height).style("border", "1px dashed");
d3.select("body").append("br")
var canvas = svg.append("g").attr("id", "canvas")
var canvas1 = svg1.append("g").attr("id", "canvas1")
var canvas2 = svg2.append("g").attr("id", "canvas2")

function draw_svg(d){draw(canvas, d, "marginal");draw(canvas1, d, "log_prob");draw(canvas2, d, "lm_marginal")}
function clear_all_canvas() {
    svg.selectAll("#canvas").selectAll("*").remove()
    svg1.selectAll("#canvas1").selectAll("*").remove()
    svg2.selectAll("#canvas2").selectAll("*").remove()
}
function reload(){
    d3.json(`../lattices/${gls[gl]}/${datasets[dataset]}/${checkpoints[checkpoint]}/${example}.json`).then(draw_svg);
}

d3.json(`../lattices/${gls[gl]}/${datasets[dataset]}/${checkpoints[checkpoint]}/${example}.json`).then(draw_svg);
function draw(canvas, data, field) {
    let node_x = d => { return d["edges"][0]["edge"]["start"] * 2 * unit * 2 + unit}
    let node_y = d => y_offset
    let node_r = d => unit * 0.3
    let node_op = d => 0.1
    data = data.map((node, i) => {return {"edges": node.map((edge, j) => {return {"edge": edge, "idx": j}}), "idx": i}})
    canvas.selectAll("g")
            .data(data)
            .enter()
            .append("g")
            .attr("id", d => `node-${d["idx"]}`)
    canvas.selectAll("g")
            .data(data)
            .append("circle")
            .attr("cx", node_x)
            .attr("cy",node_y)
            .attr("r", node_r)
            .style("fill", "blue")
            .style("opacity", node_op)
    canvas.append("g").attr("id", `node-${data.length}`)
            .append("circle")
            .attr("cx", d => data.length * 2 * unit * 2 + unit)
            .attr("cy", d => unit)
            .attr("r", unit * 0.3)
            .style("fill", "blue")
            .style("opacity", node_op)
    data.forEach(function (node, i) {
        let edge_x = d => (d["edge"]["start"] + d["edge"]["length"]-1) * 2 * unit * 2 -(d["edge"]["length"]-1) * unit * 2  + unit * 3
        let edge_y = d => (d["edge"]["length"]-1) * 2 * unit + y_offset
        let edge_r = d => unit * 0.8
        let edge_marg = d => d["edge"][field]
        var edges = canvas.select(`#node-${i}`)
            .selectAll("g")
            .data(node["edges"])
            .enter()
            .append("g")
            .attr("id", d => `node-${i}/edge-${d["edge"]["length"]}`)
        // edges.selectAll("circle")
        //     .data(node["edges"])
        //     .enter()
        //     .append("circle")
        //     .attr("cx", edge_x)
        //     .attr("cy", edge_y)
        //     .attr("r", edge_r)
        //     .style("fill", "orange")
        //     .style("opacity", edge_marg)
        edges.selectAll("path")
            .data(node["edges"])
            .enter()
            .append("path")
            .attr("d", d => curve([[node_x(node), node_y(node)],[edge_x(d), edge_y(d)],[2 * edge_x(d) - node_x(node), node_y(node)]]) )
            .attr("stroke", "orange")
            .attr("fill", 'none')
            .attr("opacity", edge_marg)
            .attr("stroke-width", d => edge_marg(d) * max_thickness)
        edges.selectAll("text")
            .data(node["edges"])
            .enter()
            .append("text")
            .text(d => d["edge"]["unit"] + `(${Math.round( +d["edge"][field] * 100 + Number.EPSILON ) / 100})`)
            .attr("x", edge_x)
            .attr("y", edge_y)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("font-size", d => `${Math.trunc(max_font * edge_marg(d))}px`)

    })
}

var zoom = d3.zoom()
    .scaleExtent([1, 10.0])
    .on("zoom", handleZoom);
function handleZoom(e) {
    // canvas.attr("transform", `translate(${e["transform"].x},0)`)
    // canvas1.attr("transform", `translate(${e["transform"].x},0)`)
    // canvas2.attr("transform", `translate(${e["transform"].x},0)`)
    canvas.attr("transform", e.transform)
    canvas1.attr("transform",  e.transform)
    canvas2.attr("transform", e.transform)
}
svg.call(zoom);
d3.select("body").append("br")
d3.select("body").append("text").text(`checkpoint: ${checkpoints[checkpoint]}`).attr("id", "ckpt_text")
d3.select("body")
    .append("input")
    .attr("id", "ckpt_bar")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", checkpoints.length-1)
    .attr("value", 0)
    .style("width", "1500px")
    .on("input", function () {
        svg.selectAll("#canvas").selectAll("*").remove()
        svg1.selectAll("#canvas1").selectAll("*").remove()
        svg2.selectAll("#canvas2").selectAll("*").remove()
        checkpoint = d3.select("#ckpt_bar").property("value")
        // canvas = svg.append("g").attr("id", "canvas")
        reload()
        d3.select("body").select("#ckpt_text").text(`checkpoint: ${checkpoints[checkpoint]}`)
    })
d3.select("body").append("br")
d3.select("body").append("text").text(`ex: ${example}`).attr("id", "ex_text")
d3.select("body")
    .append("input")
    .attr("id", "ex_bar")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", 127)
    .attr("value", 0)
    .style("width", "1500px")
    .on("input", function () {
        clear_all_canvas()
        example = d3.select("#ex_bar").property("value")
        // canvas = svg.append("g").attr("id", "canvas")
        reload()
        d3.select("body").select("#ex_text").text(`ex: ${example}`)
    })
d3.select("body").append("br")
d3.select("body").append("text").text(`dataset: ${datasets[dataset]}`).attr("id", "ds_text")
d3.select("body")
    .append("input")
    .attr("id", "ds_bar")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", datasets.length-1)
    .attr("value", 0)
    .style("width", "100px")
    .on("input", function () {
        clear_all_canvas()
        dataset = d3.select("#ds_bar").property("value")
        reload()
        d3.select("body").select("#ds_text").text(`dataset: ${datasets[dataset]}`)
    })

d3.select("body").append("br")
d3.select("body").append("text").text(`entropy_reg: ${gls[gl]}`).attr("id", "ent_text")
d3.select("body")
    .append("input")
    .attr("id", "ent_bar")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", gls.length-1)
    .attr("value", 0)
    .style("width", "100px")
    .on("input", function () {
        clear_all_canvas()
        gl = d3.select("#ent_bar").property("value")
        reload()
        d3.select("body").select("#ent_text").text(`entropy_reg: ${gls[gl]}`)
    })


console.log(gls.length)
