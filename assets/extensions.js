window.myNamespace = Object.assign({}, window.myNamespace, {  
    mySubNamespace: {  
        pointToLayer: function(feature, latlng, context) {
            return L.circleMarker(latlng);  
        },
        clickStyle: function(feature, context){
            const match = context.props.hideout && context.props.hideout.properties.name === feature.properties.name;
            if(match) return {weight:5, color:'blue', dashArray:''};
        }  
    }  
});