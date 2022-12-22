import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import it.uniroma1.lcl.babelnet.BabelNet;
import it.uniroma1.lcl.babelnet.BabelNetQuery;
import it.uniroma1.lcl.jlt.util.Language;
import it.uniroma1.lcl.jlt.util.UniversalPOS;
import it.uniroma1.lcl.kb.SynsetType;

public class ExtractSenses {

    public static void retrieveSynsetsIDs(String lemma, BabelNet bn, BufferedWriter bw, Set<String> uniqueSynsetsIDs) {
        BabelNetQuery query = new BabelNetQuery.Builder(lemma)
                .from(Language.EN)
                .POS(UniversalPOS.NOUN)
                .filterSynsets(synset -> synset.getType().equals(SynsetType.CONCEPT))
                .to(Language.EN)
                .build();

        List<String> synsetIDs = bn.getSynsets(query)
                .stream()
                .map(synset -> synset.getID().getID())
                .collect(Collectors.toList());

        uniqueSynsetsIDs.addAll(synsetIDs);

        String output = new StringBuilder()
                .append(lemma)
                .append('\t')
                .append(String.join(" ", synsetIDs))
                .append('\n')
                .toString();

        try {
            bw.write(output);
        } catch (Exception e) {
            System.err.println("Could not write " + lemma);
        }
    }

    public static void main(String[] args) {
        String inputPath = args[0];
        String outputMapPath = args[1];
        String outputUniquePath = args[2];

        BabelNet bn = BabelNet.getInstance();
        Set<String> uniqueSynsetIDs = new HashSet<>();

        List<String> lemmas;
        try (BufferedWriter bw1 = new BufferedWriter(new FileWriter(outputMapPath));
                BufferedWriter bw2 = new BufferedWriter(new FileWriter(outputUniquePath))) {
            lemmas = Files.readAllLines(Paths.get(inputPath));
            lemmas.parallelStream().forEach(lemma -> retrieveSynsetsIDs(lemma, bn, bw1, uniqueSynsetIDs));

            for (String synsetID : uniqueSynsetIDs) {
                bw2.write(synsetID + '\n');
            }

        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }

    }
}
