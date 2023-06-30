window.myNamespace = Object.assign({}, window.myNamespace, {  
    mySubNamespace: {  
        pointToLayer: function(feature, latlng, context) {
            return L.circleMarker(latlng);  
        },
        clickStyle: function(feature, context){
            return {weight:5, fillColor:'white', dashArray:''};
        }  
    }  
});