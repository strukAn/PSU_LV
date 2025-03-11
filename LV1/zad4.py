substr = "X-DSPAM-Confidence: "

def getAvgConfidence(path = ""):
    confs = []

    with open(path) as file:
        for line in file:
            line = line.strip()

            if(line[0 : len(substr)] == substr):
                confs.append(float(line[len(substr) : ]))

        print(f"Average {substr}{sum(confs) / len(confs)}") 

getAvgConfidence("mbox.txt")
getAvgConfidence("mbox-short.txt")