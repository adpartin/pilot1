load_cmeta <- function(cmeta_fullpath) {
  # Load cell line metadata
  cmeta <- read.table(cmeta_fullpath, sep="\t", header=1, na.strings=c("NA", ""))  
  cmeta <- dplyr::rename(cmeta, "CCLEName"="CCLE.name", "CellName"="Cell.line.primary.name",
                         "CellNameAliases"="Cell.line.aliases", "SitePrimary"="Site.Primary",
                         "HistSubtype1"="Hist.Subtype1", "ExpressionArrays"="Expression.arrays",
                         "SNPArrays"="SNP.arrays", "HybridCaptureSequencing"="Hybrid.Capture.Sequencing")
  return(cmeta)
}



load_dmeta <- function(dmeta_fullpath) {
  # Load drug metadata
  dmeta <- read.table(dmeta_fullpath, sep=",", header=1, na.strings=c("NA", ""))
  dmeta <- dplyr::rename(dmeta, "Drug"="Compound..code.or.generic.name.",
                         "DrugBrandName"="Compound..brand.name.", "Traget"="Target.s.",
                         "MechOfAction"="Mechanism.of.action", "HighestPhase"="Highest.Phase")
  return(dmeta)
}



load_rspdata <- function(rspdata_fullpath) {
  # Load response data
  rspdata <- read.table(rspdata_fullpath, sep=",", header=1, na.strings=c("NA", ""))
  rspdata <- dplyr::rename(rspdata, "CCLEName"="CCLE.Cell.Line.Name", "CellName"="Primary.Cell.Line.Name",
                           "Drug"="Compound", "Dose_um"="Doses..uM.", "ActivityMedian"="Activity.Data..median.",
                           "ActivitySD"="Activity.SD", "nDataPoints"="Num.Data", "EC50um"="EC50..uM.",
                           "IC50um"="IC50..uM.")
}



load_rnaseq <- function(rnaseq_fullpath) {
  message("Load data from: ", rnaseq_fullpath)
  rna <- read.table(rnaseq_fullpath, sep="\t", skip=2, header=3, na.strings=c("NA", ""), check.names=F)
  rna <- as.data.frame(rna)
  rna <- dplyr::rename(rna, "GENE_ENSG"="Name", "GENE_NAME"="Description")
  # message("dim(rna): ", dim(rna))
  
  # Rename GENE_ENSG genes (drop the ".") and reorder by GENE_ENSG
  rna$GENE_ENSG <- sapply(rna$GENE_ENSG, FUN=function(s) unlist(strsplit(as.character(s), split=".", fixed=T))[1])
  rna <- dplyr::arrange(rna, GENE_ENSG)  # rna[order(rna$GENE_ENSG),]
  
  # Extract gene names
  gene_names <- dplyr::select(rna, GENE_ENSG, GENE_NAME)
  rownames(gene_names) <- gene_names$GENE_ENSG
  
  # Drop gene name columns (keep only expression columns)
  # rna[1:2, 1:3]
  rownames(rna) <- rna$GENE_ENSG
  rna <- rna[,3:ncol(rna)]
  
  # Rename sample names (drop the parenthesis)
  colnames(rna) <- sapply(colnames(rna), FUN=function(s) unlist(strsplit(s, split=" "))[1])
  
  # library(magrittr)
  # ll <- "a-b-c-d"
  # strsplit(ll, "-") %>% sapply(extract2, 1)
  
  # Keep cell lines that were actually screened and sequenced
  # cells_screened <- unique(as.vector(rspdata$CCLEName))
  # cells_screened[1:3]
  # usecells <- intersect(cells_screened, colnames(rna))
  # message("Cells sequenced: ", ncol(rna))
  # message("Cells screened: ", length(cells_screened))
  # message("Cells screened and sequenced: ", length(usecells))
  # rna <- dplyr::select(rna, usecells)
  
  # Drop genes with all zero counts
  message("Total number of genes with all zero values in the original df: ", sum(rowSums(rna)==0))
  rna <- rna[rowSums(rna)>0,]
  
  # Update gene_names to keep the relevant genes
  # rna[1:2, 1:2]
  # gene_names[1:2, 1:2]
  gene_names <- merge(gene_names, rna, by="row.names")
  gene_names <- gene_names[,2:3] # get just the gene name cols
  # head(gene_names)
  
  # message("dim(rna): ", dim(rna))
  
  # Return list
  ll <- list("rna"=rna, "gene_names"=gene_names)
  return(ll)
}



get_gene_mappgins <- function() {
  # This function generates df of gene name mappings ('ENTREZID','ENSEMBL','SYMBOL').
  # ----------------------------------------------------------------------------------------------
  # # Example in: https://bioconductor.org/packages/release/bioc/manuals/AnnotationDbi/man/AnnotationDbi.pdf
  # # https://www.r-bloggers.com/converting-gene-names-in-r-with-annotationdbi/
  # # http://bioinfoblog.it/2015/12/tutorial-on-working-with-genomics-data-with-bioconductor-part-i/
  # AnnotationDbi::columns(org.Hs.eg.db)
  # AnnotationDbi::keytypes(org.Hs.eg.db)
  # keys <- AnnotationDbi::keys(org.Hs.eg.db, "ENTREZID")[1:5]  # get the 1st 5 possible keys (these Entrez gene IDs)
  # length(AnnotationDbi::keys(org.Hs.eg.db, "ENTREZID"))
  # # lookup gene SYMBOL and ENSEMBL ID for the 1st 5 keys
  # AnnotationDbi::select(org.Hs.eg.db, keys=keys, columns=c("SYMBOL","ENSEMBL"))
  # 
  # # get keys based on ENSEMBL
  # keyensg <- AnnotationDbi::keys(org.Hs.eg.db, keytype="ENSEMBL")
  # length(keyensg)
  # keyensg[1:5]
  # 
  # keysymbl <- AnnotationDbi::keys(org.Hs.eg.db, keytype="SYMBOL")
  # length(keysymbl)
  # keysymbl[1:5]
  # 
  # # lookup gene ENTREZID, SYMBOL, and ENSEMBL ID based on ENSEMBL IDs
  # gene_mapping <- AnnotationDbi::select(org.Hs.eg.db,
  #                                       keys=keyensg,
  #                                       columns=c("ENTREZID","ENSEMBL","SYMBOL"),
  #                                       keytype="ENSEMBL")
  # 
  # keys <- head(AnnotationDbi::keys(org.Hs.eg.db, "ENTREZID"))
  # keys
  # 
  # # get a default result (captures only the 1st element)
  # mapIds(org.Hs.eg.db, keys=keys, column='ALIAS', keytype='ENTREZID')
  # # or use a different option
  # mapIds(org.Hs.eg.db, keys=keys, column='ALIAS', keytype='ENTREZID',
  #        multiVals="ChcterList")
  # # or define your own function
  # last <- function(x) {x[[length(x)]]}
  # mapIds(org.Hs.eg.db, keys=keys, column='ALIAS', keytype='ENTREZID',
  #        multiVals=last)
  # ----------------------------------------------------------------------------------------------
  
  # lookup gene ENTREZID, SYMBOL, and ENSEMBL ID based on ENSEMBL IDs
  gene_mapping <- AnnotationDbi::select(org.Hs.eg.db,
                                        keys=AnnotationDbi::keys(org.Hs.eg.db, keytype="ENSEMBL"),
                                        columns=c("ENTREZID","ENSEMBL","SYMBOL"),
                                        keytype="ENSEMBL")
  gene_mapping <- dplyr::arrange(gene_mapping, ENSEMBL)
  print(gene_mapping[1:2,])
  # print(sapply(gene_mapping, FUN=function(x) length(unique(x))))
  
  # Keep unique gene mappings 
  idx <- !(duplicated(gene_mapping$ENSEMBL) | duplicated(gene_mapping$ENTREZID))
  gene_mapping <- gene_mapping[idx,]
  return(gene_mapping)
}



get_gene_subset <- function(rna, rna_gene_set=NULL, gene_mapping=NULL, l1k_dir=NULL) {
  # Args:
  #   rna_gene_set : "lincs", "top_exp", or else
  #   l1k_dir : base dir where L1000.txt names are located 
  #   gene_mapping : required if rna_gene_set="lincs"
  if (rna_gene_set == "lincs") {
    # L1000
    l1k <- read.table(l1k_dir, sep="\t", header=1)
    l1k <- read.table(file.path(basedir, "../../data/raw/L1000.txt"), sep="\t", header=1)
    l1k <- dplyr::rename(l1k, "ENTREZID"="ID", "SymbolLINCS"="pr_gene_symbol")
    l1k <- dplyr::select(l1k, ENTREZID, SymbolLINCS)
    l1k <- l1k[order(l1k$ENTREZID),]
    l1k <- merge(l1k, gene_mapping, by="ENTREZID")
    l1k_ <- l1k[!(l1k$SymbolLINCS==l1k$SYMBOL),]  # TODO: Note, might be some problem with gene names (?!)
    tmp_genes <- intersect(rownames(rna), l1k$ENSEMBL)
    rna <- rna[tmp_genes,]
  } else if (rna_gene_set == "top_exp") {
    # genes with highest expression (count) level
    n_top_genes <- 978 # as the number lincs genes
    x <- apply(rna, MARGIN=1, FUN=quantile, 0.25)
    y <- x[order(x, decreasing=T)][1:n_top_genes]
    rna <- rna[names(y),]
  } else {
    # Keep the original
    rna <- rna
  }
  
  return(rna)
}



plot_boxplots <- function(rna, tissuetype, filename, n=NULL) {
  if (!is.null(n)) {
    # Get a subset of samples
    set.seed(0)
    dfrna <- rna[,sample(ncol(rna), n)]
    dfrna <- rna[,1:50]
  }
  
  # ---------------------
  # Boxplot - all samples
  # ---------------------
  # https://stackoverflow.com/questions/27109347/building-a-box-plot-from-all-columns-of-data-frame-with-column-names-on-x-in-ggp?lq=1
  # https://www.data-to-viz.com/caveat/boxplot.html
  # https://www.r-graph-gallery.com/264-control-ggplot2-boxplot-colors/
  # https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html
  # https://ggplot2.tidyverse.org/reference/scale_brewer.html
  # https://plot.ly/ggplot2/facet/ --> facets
  # Reshape the log scaled count data, and create colors vector
  df1 <- stack(log2(rna+1))
  df1$clrs <- as.vector(tissuetype[df1$ind])
  # df2 <- reshape2::melt(log2(rna))
  # df1[1:3,]
  
  # http://www.cookbook-r.com/Graphs/Output_to_a_file/#ggplot2
  par(mfrow=c(1,1)); pdf(file.path(outdir, paste0(filename, ".pdf")), width=150)
  print(
    ggplot(df1, aes(x=ind, y=values)) +  
    # geom_violin(width=1.4) +
    geom_boxplot(aes(fill=factor(clrs)), alpha=0.9) +
    # geom_jitter(color="grey", size=0.7, alpha=0.5) +
    # scale_fill_viridis(discrete=T, option="magma") +
    ggtitle("CCLE samples") + ##ggtitle("CCLE samples (raw)") +
    xlab("") +
    ##ylab("log2(cnts+1)") +
    theme_dark() + 
    scale_colour_brewer(palette="Set1") +
    # theme(legend.position="bottom",
    #       legend.direction="horizontal",
    #       plot.title=element_text(size=11),
    #       axis.text.x=element_text(angle=60, hjust=1))
    theme(axis.text.x=element_text(angle=60, hjust=1))
  )
  # facet_wrap(~clrs)
  dev.off()
}
