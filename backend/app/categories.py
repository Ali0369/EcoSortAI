"""Static, human-written category information (NOT model output).

Guidance is deliberately general: recycling rules differ by location, so every card says to check
local rules.  Colors are the bin colors sampled from the supplied Figma design.
"""

STATION_ORDER = ["paper", "plastic", "glass", "cardboard", "metal", "trash"]   # bin order in the sorting scene
CARD_ORDER = ["plastic", "paper", "cardboard", "glass", "metal", "trash"]      # order of the category cards

CATEGORIES = {
    "plastic": {
        "label": "Plastic", "recyclable": True, "bin_front": "#204739", "bin_side": "#122E25", "bin_rim": "#D4EF82",
        "headline": "Empty. Rinse. Recycle.",
        "guidance": "Remove any liquid and check your local rules for caps.",
        "summary": "Bottles, tubs and packaging made from plastic. Many kerbside programs accept bottles and tubs but not thin films or bags.",
        "steps": [("Empty it", "Pour out any remaining liquid."), ("Give it a quick rinse", "No need to make it spotless."), ("Check caps locally", "Rules vary by collection service.")],
    },
    "paper": {
        "label": "Paper", "recyclable": True, "bin_front": "#A5B4A0", "bin_side": "#748B79", "bin_rim": "#DCE7DF",
        "headline": "Keep it clean and dry.",
        "guidance": "Clean, dry paper is recyclable. Greasy or wet paper usually is not.",
        "summary": "Office paper, newspapers, magazines and similar sheets.",
        "steps": [("Keep it dry", "Wet paper is hard to recycle."), ("Remove plastic wrapping", "Take out any plastic sleeves."), ("Skip greasy paper", "Food-soiled paper often goes in general waste.")],
    },
    "cardboard": {
        "label": "Cardboard", "recyclable": True, "bin_front": "#D3C4A6", "bin_side": "#A99A7F", "bin_rim": "#ECE2D1",
        "headline": "Flatten it first.",
        "guidance": "Flatten boxes and keep them dry. Remove excess tape if you can.",
        "summary": "Shipping boxes and corrugated or card packaging.",
        "steps": [("Flatten it", "Save space in the bin."), ("Keep it dry", "Wet cardboard loses its value."), ("Remove food residue", "Greasy parts (like pizza-box bottoms) may not be accepted.")],
    },
    "glass": {
        "label": "Glass", "recyclable": True, "bin_front": "#B6C9CB", "bin_side": "#80989C", "bin_rim": "#DFE9ED",
        "headline": "Rinse and recycle.",
        "guidance": "Rinse bottles and jars. Ceramics, mirrors and window glass are usually not accepted.",
        "summary": "Glass bottles and jars. Other glass types are often processed separately.",
        "steps": [("Empty it", "Pour out any contents."), ("Give it a rinse", "A quick rinse is enough."), ("Check lids locally", "Some services want lids on, others off.")],
    },
    "metal": {
        "label": "Metal", "recyclable": True, "bin_front": "#D9DDD8", "bin_side": "#929F9B", "bin_rim": "#E5E8E5",
        "headline": "Empty and recycle.",
        "guidance": "Rinse cans and tins. Only recycle aerosol cans that are completely empty, and follow local rules.",
        "summary": "Food cans, drink cans and other metal packaging.",
        "steps": [("Empty it", "Remove any food or liquid."), ("Give it a rinse", "Keep it simple."), ("Check special items", "Aerosols and paint tins often have separate rules.")],
    },
    "trash": {
        "label": "Trash", "recyclable": False, "bin_front": "#48534F", "bin_side": "#272F2D", "bin_rim": "#BCCDC2",
        "headline": "General waste.",
        "guidance": "This item usually goes in general waste. Batteries, electronics and hazardous items need special drop-off.",
        "summary": "Items that are not accepted in most kerbside recycling.",
        "steps": [("Check for special items", "Batteries and electronics need dedicated collection."), ("Reuse if you can", "Repair or donate when possible."), ("Place in general waste", "Bag it and bin it.")],
    },
}
