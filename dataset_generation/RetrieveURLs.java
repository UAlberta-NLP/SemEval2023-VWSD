import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

import it.uniroma1.lcl.babelnet.BabelNet;
import it.uniroma1.lcl.babelnet.BabelSynset;
import it.uniroma1.lcl.babelnet.BabelSynsetID;
import it.uniroma1.lcl.babelnet.InvalidSynsetIDException;

public class RetrieveURLs {

    public static void retrieveImages(String synsetID, BabelNet bn, BufferedWriter writer) {

        BabelSynsetID goldSynsetID = null;
        BabelSynset goldSynset = null;

        try {
            goldSynsetID = new BabelSynsetID(synsetID);
            goldSynset = goldSynsetID.toSynset();

            if (goldSynset == null)
                return;
        } catch (InvalidSynsetIDException e) {
            return;
        }

        List<String> urls = goldSynset.getImages()
                .stream()
                .map(babelImage -> babelImage.getURL())
                .collect(Collectors.toList());

        if (urls.isEmpty())
            return;

        String output = new StringBuilder()
                .append(synsetID)
                .append('\t')
                .append(String.join(" ", urls))
                .append('\n')
                .toString();

        try {
            writer.write(output);
        } catch (IOException e) {
            System.err.println("Could not write urls for " + synsetID);
        }
    }

    public static void main(String[] args) {
        String inputPath = args[0];
        String outputPath = args[1];

        BabelNet bn = BabelNet.getInstance();

        List<String> synsetIDs;
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath))) {
            synsetIDs = Files.readAllLines(Paths.get(inputPath));

            synsetIDs.stream().forEach(id -> retrieveImages(id, bn, bw));
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Could not read the line");
        }

    }
}