with open('morph.tet6', 'r') as fh:
    text_split = fh.read().split("\n4 ")

with open('points.txt', 'w') as fh:
    fh.write(text_split[0])

with open('table.txt', 'w') as fh:
    # If you know that the keyword only appears once
    # you can changes this to fh.write(text_split[1])
    temp="\n4 ".join(text_split[1:])
    fh.write("4 "+temp)
