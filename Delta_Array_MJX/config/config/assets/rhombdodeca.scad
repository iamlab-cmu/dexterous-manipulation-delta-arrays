////////////////////////////////////////////////
// PARAMETERS
////////////////////////////////////////////////
L = 40;       // Base rhombic dodecahedron 'height' in the original design
w = 0.5;      // Fraction of face to hollow
d = 16.5;     // Distance factor
f = 1.03;     // Fudge fit for subtractions
n = 25;       // Number of links to get ~300 mm chain
s = w/2 + 0.5; // Used inside openRhomb()
heightScale = 15 / L;  // So final chain height = 15 mm

////////////////////////////////////////////////
// MAIN CALL
////////////////////////////////////////////////
scale([heightScale, heightScale, heightScale]) {
  chain(n);
}

// Optional bounding box to visualize 300×(some width) on your bed
%translate([-150, -10, -0.5])   // shift so bounding box is somewhat centered
%cube([300, 30, 1]);            // 300 mm long, 30 mm wide “reference plane”

////////////////////////////////////////////////
// MODULES
////////////////////////////////////////////////

// Generate a linear chain of n links
module chain(n){
  // Shift it so it's roughly centered near x=0
  // so you can see it easily in your preview
  translate([-(n-1)*d + d, 0, 0])
  for(i = [1:n]) {
    translate([(i-1)*2*d, 0, 0]) 
      openRhomb();
  }
}

// Single link with open faces
module openRhomb(){
  // "render()" to force final geometry
  render() 
  rotate([0,0,acos(1/3)/2])
  translate([0,0,L/2])
  rotate([-45,0,90])
  difference(){
    rhombDodeca(1);
    
    // Subtractions to hollow it out and create interlock gaps
    rhombDodeca(s);
    rotate([90,0,0]) rhombDodeca(s);
    rotate([0,90,0]) rhombDodeca(s);
  }
}

// Rhombic dodecahedron
module rhombDodeca(scaleFactor){
  intersection(){
    // Each cube is oriented differently and scaled to create the rhombic shape
    rotate([0,0,45])  cube(L*[1/scaleFactor,1/scaleFactor,2], center=true);
    rotate([0,45,0])  cube(L*[scaleFactor,2,scaleFactor],     center=true);
    rotate([45,0,0])  cube(L*[2,scaleFactor,scaleFactor],     center=true);
  }
}
