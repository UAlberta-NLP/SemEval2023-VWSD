import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import it.uniroma1.lcl.babelnet.BabelNet;
import it.uniroma1.lcl.babelnet.BabelNetQuery;
import it.uniroma1.lcl.babelnet.BabelSynset;
import it.uniroma1.lcl.babelnet.BabelSynsetID;
import it.uniroma1.lcl.babelnet.InvalidSynsetIDException;
import it.uniroma1.lcl.babelnet.data.BabelImage;
import it.uniroma1.lcl.jlt.util.Language;
import it.uniroma1.lcl.jlt.util.UniversalPOS;
import it.uniroma1.lcl.kb.SynsetType;

public class RetrieveImages {

    public static void retrieveImages(String lemma, String goldSenseKey, BabelNet bn, BufferedWriter writer)
            throws IOException {
        BabelNetQuery query = new BabelNetQuery.Builder(lemma)
                .from(Language.EN)
                .POS(UniversalPOS.NOUN)
                .filterSynsets(synset -> synset.getType().equals(SynsetType.CONCEPT))
                .to(Language.EN)
                .build();

        List<BabelSynset> synsets = bn.getSynsets(query);

        List<String> candidateURLs = synsets.stream().parallel()
                .filter(s -> !s.getID().getID().equals(goldSenseKey))
                .map(s -> s.getMainImage())
                .filter(img -> img.isPresent())
                .map(img -> img.get().getURL())
                .collect(Collectors.toList());

        BabelSynsetID goldSynsetID = null;
        BabelSynset goldSynset = null;

        try {
            goldSynsetID = new BabelSynsetID(goldSenseKey);
            goldSynset = goldSynsetID.toSynset();

            if (goldSynset == null)
                return;
        } catch (InvalidSynsetIDException e) {
            return;
        }

        Optional<BabelImage> goldImage = goldSynset.getMainImage();

        if (!goldImage.isPresent() | candidateURLs.isEmpty())
            return;

        String goldURL = goldImage.get().getURL();

        String output = new StringBuilder()
                .append(lemma)
                .append('\t')
                .append(goldSenseKey)
                .append('\t')
                .append(goldURL)
                .append('\t')
                .append(String.join(" ", candidateURLs))
                .append('\n')
                .toString();

        writer.write(output);

    }

    public static void main(String[] args) {
        String inputPath = args[0];
        String outputPath = args[1];

        BabelNet bn = BabelNet.getInstance();

        try (BufferedReader br = new BufferedReader(new FileReader(inputPath));
                BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] lemmaSense = line.split("\t");
                String lemma = lemmaSense[0];
                String goldSenseKey = lemmaSense[1];

                retrieveImages(lemma, goldSenseKey, bn, bw);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("Could not read the line");
        }

    }
}