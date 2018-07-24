
function clear_div(div_name){
    div = document.getElementById(div_name);
    while(div.firstChild) {
        div.removeChild(div.firstChild);
    }
}

function load_image(basename, type, ant1, ant2){
    // Create image object for a given baseline and type
    
    ////////////////////////////////////////////
    // FIXME: Testing
    ant1 = 11;
    ant2 = 12;
    test_dir = "/home/phil/Desktop/plotty/";
    // END FIXME
    ////////////////////////////////////////////
    
    var img = document.createElement("img");
    img.src = test_dir + basename + "." + type + "." + ant1 + "." + ant2 + ".png";
    img.id = img.src;
    return img;
}

function list_baselines(){
    // List all available baselines
    clear_div("flow");
    
    var divList = document.createElement("div");
    divList.className = "bl-list";
    
    // List title
    var hTitle = document.createElement("h2");
    hTitle.appendChild(document.createTextNode("Baselines"));
    
    // Get data structure
    var dat = new Data();
    
    // Create list of redundant groups
    lstRedGroups = document.createElement("ul");
    
    // Populate list
    for(i=0; i < dat.red_bls.length; i++){
        
        // Create sub-list for this redundant group
        liRedGroups = document.createElement("li");
        var hGroup = document.createElement("h3");
        hGroup.appendChild(document.createTextNode("Redundant group " + i));
        lstBls = document.createElement("ul");
        
        // Populate sub-list with redundant baselines
        for(j=0; j < dat.red_bls[i].length; j++){
            var li = document.createElement("li");
            var ant1 = dat.red_bls[i][j][0];
            var ant2 = dat.red_bls[i][j][1];
            li.appendChild(
                document.createTextNode( "Baseline (" + ant1 + ", " + ant2 + ")" ) );
            lstBls.appendChild(li);
        } // end loop over bls
        
        liRedGroups.appendChild(hGroup);
        liRedGroups.appendChild(lstBls);
        lstRedGroups.appendChild(liRedGroups);
    } // end loop over redundant groups
    
    // Attach to 
    divList.appendChild(hTitle);
    divList.appendChild(lstRedGroups);
    document.getElementById("flow").appendChild(divList);
}


function grid_red_bls(basename, type){
    // List all available baselines
    var divRedGroups = document.createElement("div");
    divRedGroups.className = "red-groups";
    
    // Grid title
    var hTitle = document.createElement("h2");
    hTitle.appendChild(
        document.createTextNode("Grid:" + basename + " " + type) );
    
    // Get data structure
    var dat = new Data();
    
    // Populate grid
    for(i=0; i < dat.red_bls.length; i++){
        
        // Create div for this redundant group
        var divGroup = document.createElement("div");
        divGroup.className = "red-group";
        divGroup.id = i;
        var hGroup = document.createElement("h3");
        hGroup.appendChild(document.createTextNode("Redundant group " + i));
        
        // Populate div with plot for each baseline
        var divGroupImgs = document.createElement("div");
        divGroupImgs.className = "red-group-imgs";
        divGroupImgs.id = i;
        for(j=0; j < dat.red_bls[i].length; j++){
            
            // Load image
            img = load_image(basename, type, 
                             dat.red_bls[i][j][0], 
                             dat.red_bls[i][j][1]);
            
            img.href = "#img" + j;
            img.style.width = "20%";
            
            divGroupImgs.appendChild(img);
        } // end loop over bls
        
        divGroup.appendChild(hGroup);
        divGroup.appendChild(divGroupImgs);
        divRedGroups.appendChild(divGroup);
    } // end loop over redundant groups
    
    // Attach to document
    document.getElementById("flow").appendChild(divRedGroups);
}


function add_plot(basename, type, ant1, ant2){
    // Create a new div with a plot and basic info
    
    var divPlot = document.createElement("div");
    divPlot.className = "plot";
    
    // Plot title
    var hTitle = document.createElement("h2");
    var txtTitle = document.createTextNode(type);
    hTitle.appendChild(txtTitle);
    
    // Load plot image and put in container
    var divInner = document.createElement("div");
    divInner.setAttribute("class", "img-zoom-container");
    divInner.style.position = "relative";
    
    var imgPlot = load_image(basename, type, ant1, ant2);
    //var imgPlot = document.createElement("img");
    //imgPlot.src = basename + "." + type + "." + ant1 + "." + ant2 + ".png";
    //imgPlot.id = imgPlot.src;
    imgPlot.style.width = "40%";
    
    // Don't attach the zoom capability until the image has loaded
    function attach_zoom(){ imageZoom(imgPlot.id, "zoom-window"); }
    imgPlot.onmouseover = attach_zoom;
    
    // Build div  
    divPlot.appendChild(hTitle);
    divInner.appendChild(imgPlot);
    divPlot.appendChild(divInner);
    
    // Append div to main document
    document.getElementById("flow").appendChild(divPlot);
} // end function


function load_baseline(ant1, ant2){
    // Load summary data and plots for a given baseline
    
    clear_div("flow");
    flow = document.getElementById("flow");
    var bl_title = document.createElement("h2");
    bl_title.id = "blname";
    bl_title.innerHTML = "Baseline (" + ant1 + ", " + ant2 + ")";
    flow.appendChild(bl_title);
    
    add_plot("zen.grp1.of1.xx.LST.1.71736", "uvOCRSLTF.XX", ant1, ant2);
    add_plot("zen.grp1.of1.yy.LST.1.71736", "uvOCRSLTF.YY", ant1, ant2);
    add_plot("zen.grp1.of1.pI.LST.1.71736", "uvOCRSLTF.pI", ant1, ant2);
    add_plot("zen.grp1.of1.xx.tavg", "uvOCRSL.XX", ant1, ant2);
    
} // end load image

// Code adapted from: https://www.w3schools.com/howto/howto_js_image_zoom.asp
function imageZoom(imgID, resultID) {
  
  var img, lens, result, cx, cy;
  img = document.getElementById(imgID);
  result = document.getElementById(resultID);
  
  // Check if this image has already been loaded into the zoom window
  if (result.style.backgroundImage == "url(\"" + img.src + "\")"){
    return;
  }

  // Create lens (if not already created)
  lens = document.getElementById("lens-rectangle");
  if (lens == null){
      lens = document.createElement("div");
      lens.setAttribute("class", "img-zoom-lens");
      lens.setAttribute("id", "lens-rectangle");
  }

  // Insert lens:
  img.parentElement.insertBefore(lens, img);

  // Calculate the ratio between result DIV and lens
  cx = result.offsetWidth / lens.offsetWidth;
  cy = result.offsetHeight / lens.offsetHeight;

  // Set background properties for the result DIV
  result.style.backgroundImage = "url('" + img.src + "')";
  result.style.backgroundSize = (img.width * cx) + "px " + (img.height * cy) + "px";

  // Execute a function when someone moves the cursor over the image, or the lens
  lens.addEventListener("mousemove", moveLens);
  img.addEventListener("mousemove", moveLens);
  lens.addEventListener("touchmove", moveLens);
  img.addEventListener("touchmove", moveLens);
  
  function moveLens(e) {
    var pos, x, y;
    // Prevent any other actions that may occur when moving over the image
    e.preventDefault();
    
    // Get the cursor's x and y positions
    pos = getCursorPos(e);
    
    // Make sure lens is visible
    lens.style.visible = true;
    
    // Calculate the position of the lens
    x = pos.x - (lens.offsetWidth / 2);
    y = pos.y - (lens.offsetHeight / 2);
    
    // Prevent the lens from being positioned outside the image
    if (x > img.width - lens.offsetWidth) {x = img.width - lens.offsetWidth;}
    if (x < 0) {x = 0;}
    if (y > img.height - lens.offsetHeight) {y = img.height - lens.offsetHeight;}
    if (y < 0) {y = 0;}
    
    // Set the position of the lens
    lens.style.left = x + "px";
    lens.style.top = y + "px";
    
    // Display what the lens "sees"
    result.style.backgroundPosition = "-" + (x * cx) + "px -" + (y * cy) + "px";
  }
  
  function getCursorPos(e) {
    var a, x = 0, y = 0;
    e = e || window.event;
    // Get the x and y positions of the image:
    a = img.getBoundingClientRect();
    
    // Calculate the cursor's x and y coordinates, relative to the image:
    x = e.pageX - a.left;
    y = e.pageY - a.top;
    
    // Consider any page scrolling:
    x = x - window.pageXOffset;
    y = y - window.pageYOffset;
    return {x : x, y : y};
  }
} // end function
