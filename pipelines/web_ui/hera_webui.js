
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

    function raise_image_error(){
        console.log("Image load error.");
    } // FIXME

    var img = document.createElement("img");
    img.src = test_dir + basename + "." + type + "." + ant1 + "." + ant2 + ".png";
    img.id = img.src;
    img.onerror = raise_image_error;
    return img;
}

//function add_select(opts, opt_names, callback){
//    // Add a dropdown menu to the document
//    var sel = document.createElement("select");
//
//    // Add options from list
//    for (i=0; i < opts.length; i++){
//        var opt = document.createElement('option');
//        opt.value = opts[i];
//        opt.innerHTML = opt_names[i];
//        sel.add(opt);
//    }
//
//    // Function to return selected value to a user-defined callback function
//    function select_callback(){
//        var val = drop.value;
//        callback(val);
//    }
//
//    // Attach callback
//    sel.addEventListener("change", select_callback);
//    return sel;
//}

function select_callback(dropdown, item){
    mark_dropdown_selected(dropdown, item);

    // Display a grid of plots
    dropdown_grid();
}

function populate_dropdown(dropdown, opts, opt_names, callback_fn){
    // Add elements to dropdown list

    // Get the dropdown and clear it, and mark it as enabled
    sel = clear_dropdown(dropdown, true);

    // Add default item
    var item = document.createElement('a');
    item.innerHTML = "&mdash;";
    item.className = "w3-bar-item w3-button";
    item.setAttribute("href", "javascript:" + callback_fn + "('"
                                + dropdown + "', 'none');");
    item.id = dropdown + "-none";
    sel.appendChild(item);

    // Add options from list
    for (i=0; i < opts.length; i++){
        var item = document.createElement('a');
        item.innerHTML = opt_names[i];
        item.className = "w3-bar-item w3-button";
        item.setAttribute("href", "javascript:" + callback_fn + "('"
                                + dropdown + "', '" + opts[i] + "');");
        item.id = dropdown + "-" + opts[i];
        sel.appendChild(item);
    }
}

function clear_dropdown(dropdown, enabled){
    // Disable a dropdown menu

    // Get select and clear it
    var sel = document.getElementById(dropdown + "-select");
    while(sel.firstChild) { sel.removeChild(sel.firstChild); }

    // Grey-out the dropdown button and clear the info box if disabled
    var btn = document.getElementById(dropdown + "-button");
    var info = document.getElementById(dropdown + "-status");
    if (enabled){
        btn.style.opacity = 1.0;
        info.style.opacity = 1.0;
    }else{
        btn.style.opacity = 0.5;
        info.style.opacity = 0.5;
    }
    info.innerHTML = "&mdash;";

    // Return the dropdown
    return sel;
}

function mark_dropdown_selected(dropdown, option){
    // Mark a given argument of a dropdown as selected

    // Loop over items in dropdown and make sure they're not highlighted
    var list_elements = document.getElementById(dropdown + "-select").children;
    for(i=0; i < list_elements.length; i++){
        list_elements[i].style.fontWeight = 'normal';
    }

    // Highlight selected item
    var item = document.getElementById(dropdown + "-" + option);
    item.style.fontWeight = 'bold';

    // Change status indicator
    var status = document.getElementById(dropdown + "-status");
    status.innerHTML = item.innerHTML;
}

function move_next_lst(evt){
    // Go back or forward in LST when left/right key is pressed
    var dat = new Data();
    move_next_item(evt, "lst", dat.lsts);
}

function move_next_item(evt, dropdown, array){
    // Catch keypress events and go back or forward in some variable
    var e = evt || window.event;
    var c = e.keyCode;

    // Left or right key
    if ((c == 39) || (c == 37)){

        // Get currently selected item
        var cur = get_selected_item(dropdown);
        if (cur[0] == -1){ return; }
        var i = cur[0] - 1;

        // Increment/decrement counter
        if (c == 39){
            if (i >= array.length - 1){ i = 0; }else{ i++; } // Right key
        }else{
            if (i == 0){ i = array.length - 1; }else{ i--; } // Left key
        }

        // Select new item and return
        mark_dropdown_selected(dropdown, array[i]);
        dropdown_grid();
        return;

    } // end left/right check

    /*
    if ((c == 38) || (c == 40)){

        // Get currently selected blgroup
        var redgrp = get_selected_item("redgrp");
        if (redgrp[0] == -1){ return; }
        var i = redgrp[0] - 1;

        // Increment/decrement counter
        if (c == 38){
            if (i >= dat.red_bls.length - 1){ i = 0; }else{ i++; } // Up key
        }else{
            if (i == 0){ i = dat.lsts.length - 1; }else{ i--; } // Down key
        }

        // Select new red. group and return
        mark_dropdown_selected("redgrp", dat.red_bls[i]);
        dropdown_grid();
        return;
    } // end up/down check
    */
}

/*
<a href="#" onclick="select_pipeline_stage();"
   class="w3-bar-item w3-button">LST-binned data</a>
<a href="#" onclick="select_pipeline_stage();"
   class="w3-bar-item w3-button">RFI flagging</a>
<a href="#" onclick="select_pipeline_stage();"
   class="w3-bar-item w3-button">Time averaging</a>
<a href="#" onclick="select_pipeline_stage();"
   class="w3-bar-item w3-button">Pseudo-Stokes</a>
<a href="#" onclick="select_pipeline_stage();"
   class="w3-bar-item w3-button">Delay filtering</a>
*/

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

    // Build dropdown select list
    populate_dropdown("pipeline", dat.prefixes, dat.prefix_names, "select_callback");
    populate_dropdown("lst", dat.lsts, dat.lsts, "select_callback");
    populate_dropdown("pol", dat.pols, dat.pols, "select_callback");
    populate_dropdown("redgrp", dat.red_bls, dat.red_bls, "select_callback");

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

    // Attach to document
    divList.appendChild(hTitle);
    //divList.appendChild(sel);
    divList.appendChild(lstRedGroups);
    document.getElementById("flow").appendChild(divList);
}

function get_selected_item(dropdown){
    // Get the item selected in a drop-down menu. Returns 2-element array,
    // [index, item].

    var items = document.getElementById(dropdown + "-select").children;
    var selected = dropdown + "-none";
    var idx = -1;
    for(i=0; i < items.length; i++){
        if (items[i].style.fontWeight == 'bold'){
            selected = items[i].id;
            idx = i;
        }
    }

    // Parse the selected value
    selected = selected.slice(dropdown.length + 1);
    if (selected == "none"){ idx = -1; }
    return [idx, selected];
}


function dropdown_grid(){
    // Show a grid of plots according to which drop-down options are selected
    // grid_red_bls("zen.grp1.of1.xx.LST.1.71736", "uvOCRSLTF.XX");

    // Get selected items (idx, name)
    var pipe = get_selected_item("pipeline");
    var lst = get_selected_item("lst");
    var redgrp = get_selected_item("redgrp");
    var pol = get_selected_item("pol");

    // FIXME: Should be different cases depending on what's selected
    if ((pipe[0] != -1) && (lst[0] != -1) && (pol[0] != -1)){
        var dat = new Data();

        // Special case for xx and yy polarization strings
        var pol1 = pol[1], pol2 = pol[1];
        if ((pol1.toLowerCase() == 'xx') || (pol1.toLowerCase() == 'yy')){
            pol1 = pol1.toLowerCase(); pol2 = pol2.toUpperCase();
        }

        // Construct basename and type
        var basename = dat.root + pol1 + ".LST." + lst[1];
        var type = pipe[1] + "." + pol2;

        // Show grid of redundant baselines (or just one, if requested)
        if (redgrp[0] == -1){
            grid_red_bls(basename, type, "all"); // show all
        }else{
            grid_red_bls(basename, type, redgrp[0] - 1);
        }
        return;
    }

    // By default, show nothing except a hint
    clear_div("flow");
    var msg_box = message_box("Select some options", "Please select some combination of LST, polarization, baseline group, and pipeline stage.");
    document.getElementById("flow").appendChild(msg_box);
}

function message_box(title, text){
    // Show a message box

    // Container
    var box = document.createElement("div");
    box.className = "w3-panel w3-padding-16 w3-blue-gray";
    box.style.width = "70%";

    // Title
    var hTitle = document.createElement("h3");
    hTitle.appendChild( document.createTextNode(title) );
    box.appendChild(hTitle);

    // Text
    var ptext = document.createElement("p");
    ptext.appendChild( document.createTextNode(text) );
    box.appendChild(ptext);

    return box;
}


function grid_red_bls(basename, type, idx){
    // List all available baselines (or only a particular index if specified)

    clear_div("flow");

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
        if ((idx != "all") && (i != idx)){ continue; }

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

/*
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

} // end load baseline
*/



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
