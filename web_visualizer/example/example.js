var scale = 30;
var unit = 1 * scale;
var max_thickness = 1 * scale;
var max_font = 2 * scale;
var width = 50 * unit;
var height = 30 * unit;
const curve = d3.line().curve(d3.curveNatural);

var svg = d3.select("body").append("svg").attr("width", width).attr("height", height).style("border", "1px dashed");
d3.json("lattice.json").then(function (d){draw(d)});
function draw(data) {
    let node_x = d => d["idx"] * 2 * unit * 2 + unit
    let node_y = d => unit
    let node_r = d => unit * 0.3
    let node_op = d => 0.1
    data = data.map((node, i) => {return {"edges": node.map((edge, j) => {return {"edge": edge, "idx": j}}), "idx": i}})
    console.log(data)
    svg.selectAll("g")
            .data(data)
            .enter()
            .append("g")
            .attr("id", d => `node-${d["idx"]}`)
    svg.selectAll("g")
            .data(data)
            .append("circle")
            .attr("cx", node_x)
            .attr("cy",node_y)
            .attr("r", node_r)
            .style("fill", "blue")
            .style("opacity", node_op)
    svg.append("g").attr("id", `node-${data.length}`)
            .append("circle")
            .attr("cx", d => data.length * 2 * unit * 2 + unit)
            .attr("cy", d => unit)
            .attr("r", unit * 0.3)
            .style("fill", "blue")
            .style("opacity", node_op)
    data.forEach(function (node, i) {
        let edge_x = d => (i + d["edge"][2]-1) * 2 * unit * 2 -(d["edge"][2]-1) * unit * 2  + unit * 3
        let edge_y = d => (d["edge"][2]-1) * 2 * unit + unit
        let edge_r = d => unit * 0.8
        let edge_marg = d => d["edge"][1]
        var edges = svg.select(`#node-${i}`)
            .selectAll("g")
            .data(node["edges"])
            .enter()
            .append("g")
            .attr("id", d => `node-${i}/edge-${d["edge"][2]}`)
        // edges.selectAll("circle")
        //     .data(node["edges"])
        //     .enter()
        //     .append("circle")
        //     .attr("cx", edge_x)
        //     .attr("cy", edge_y)
        //     .attr("r", edge_r)
        //     .style("fill", "orange")
        //     .style("opacity", edge_marg)
        edges.selectAll("text")
            .data(node["edges"])
            .enter()
            .append("text")
            .text(d => d["edge"][0])
            .attr("x", edge_x)
            .attr("y", edge_y)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("font-size", d => `${Math.trunc(max_font * edge_marg(d))}px`)
        edges.selectAll("path")
            .data(node["edges"])
            .enter()
            .append("path")
            .attr("d", d => curve([[node_x(node), node_y(node)],[edge_x(d), edge_y(d)],[2 * edge_x(d) - node_x(node), node_y(node)]]) )
            .attr("stroke", "orange")
            .attr("fill", 'none')
            .attr("opacity", edge_marg)
            .attr("stroke-width", d => edge_marg(d) * max_thickness)

    })
}

console.log("hello")