from fontTools.ttLib import TTFont

# Load the downloaded font file
font = TTFont('p151.woff2')  # Use 'p1.ttf' if you've converted it to TTF

# Extract the character map (cmap)
cmap = font['cmap'].getBestCmap()

# Display the mappings
for codepoint, glyph_name in cmap.items():
    print(f"Unicode: {hex(codepoint)}, Glyph Name: {glyph_name}")
