# L1000 genes: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL20573
rm(list=ls())
library(tictoc)
library(glue)
library(dplyr)
library(ggplot2)
library(gplots)
library(pheatmap)
library(org.Hs.eg.db)
library(reshape2)
library(tibble)
library(DESeq2)

# Choose what data in terms of rna-seq genes to process and save
rna_gene_set <- "lincs";  save_data_name <- "lincs" # landmark genes
# rna_gene_set <- "top_exp_1k";  save_data_name <- "top_genes"  # most expressed genes ~1000
# rna_gene_set <- "full";  save_data_name <- "full_gene_set"  # full set of genes after filtering

# Set path to source file location
# Sessop --> Set Working Directory --> To Source File Location
# basedir <- "/Users/apartin/Dropbox/work/pilot1/cell-line"
basedir <- getwd()

# Outdir
outdir <- file.path(basedir, "../../data/processed/from_ccle_org")
if (!dir.exists(outdir)) {dir.create(outdir)}
# list.files()

# Path to the original data
datadir <- "/Users/apartin/work/jdacs/cell-line-data/ccle/from_broad_institute"
datadir_molecular <- file.path(datadir, "current_data_11-08-2018")
datadir_cmeta <- file.path(datadir, "cell_line_annotations")
datadir_dmeta <- file.path(datadir, "pharmacological_profiling")



# ==============================================================
#   Load cell metadata
# ==============================================================
cmeta <- read.table(file.path(datadir_cmeta, "CCLE_sample_info_file_2012-10-18.txt"),
                    sep="\t", header=1, na.strings=c("NA", ""))
# dim(cmeta)
# colnames(cmeta)
# cellmeta[1:3,]
cmeta <- dplyr::rename(cmeta, "CCLEName"="CCLE.name", "CellName"="Cell.line.primary.name",
                       "CellNameAliases"="Cell.line.aliases", "SitePrimary"="Site.Primary",
                       "HistSubtype1"="Hist.Subtype1", "ExpressionArrays"="Expression.arrays",
                       "SNPArrays"="SNP.arrays", "HybridCaptureSequencing"="Hybrid.Capture.Sequencing")
# sapply(cmeta, FUN=function(x) length(unique(x)))
# apply(cmeta, MARGIN=2, FUN=function(x) length(unique(x)))



# ==============================================================
#   Load drug metadata
# ==============================================================
dmeta <- read.table(file.path(datadir_dmeta, "CCLE_NP24.2009_profiling_2012.02.20.csv"),
                    sep=",", header=1, na.strings=c("NA", ""))
# dim(dmeta)
# colnames(dmeta)
# dmeta[1:3,]
dmeta <- dplyr::rename(dmeta, "Drug"="Compound..code.or.generic.name.",
                       "DrugBrandName"="Compound..brand.name.", "Traget"="Target.s.",
                       "MechOfAction"="Mechanism.of.action", "HighestPhase"="Highest.Phase")
# apply(dmeta, MARGIN=2, FUN=function(x) length(unique(x)))



# ==============================================================
#   Load response data
# ==============================================================
rspdata <- read.table(file.path(datadir_dmeta, "CCLE_NP24.2009_Drug_data_2015.02.24.csv"),
                      sep=",", header=1, na.strings=c("NA", ""))
# dim(rspdata)
# colnames(rspdata)
# rspdata[1:2,]
rspdata <- dplyr::rename(rspdata, "CCLEName"="CCLE.Cell.Line.Name", "CellName"="Primary.Cell.Line.Name",
                         "Drug"="Compound", "Dose_um"="Doses..uM.", "ActivityMedian"="Activity.Data..median.",
                         "ActivitySD"="Activity.SD", "nDataPoints"="Num.Data", "EC50um"="EC50..uM.",
                         "IC50um"="IC50..uM.")
# apply(dplyr::select(rspdata, CCLEName, CellName, Drug), MARGIN=2,
#       FUN=function(x) length(unique(x)))

# colSums(is.na(rspdata))



# ==============================================================
#   Load RNA-Seq
# ==============================================================
rna <- read.table(file.path(datadir_molecular, "CCLE_DepMap_18q3_RNAseq_reads_20180718.gct.txt"),
                  sep="\t", skip=2, header=3, na.strings=c("NA", ""), check.names=F)
dim(rna)
colnames(rna)[1:4]
rna[1:2, 1:4]
# rna <- dplyr::rename(rna, "ENSG"="Name", "GeneName"="Description")
rownames(rna) <- rna$Name
rna <- rna[,3:ncol(rna)]

# Rename ENSG genes (drop the ".") and reorder by ENSG
# rna$ENSG <- sapply(rna$ENSG, FUN=function(s) unlist(strsplit(as.character(s), split=".", fixed=T))[1])
# rna <- dplyr::arrange(rna, ENSG)
rownames(rna) <- sapply(rownames(rna), FUN=function(s) unlist(strsplit(as.character(s), split=".", fixed=T))[1])
rna <- rna[order(rownames(rna)),]

# Rename samples (drop the parenthesis)
# colnames(rna)[3:ncol(rna)] <- sapply(colnames(rna)[3:ncol(rna)],
#                                      FUN=function(s) unlist(strsplit(s, split=" "))[1])
colnames(rna) <- sapply(colnames(rna), FUN=function(s) unlist(strsplit(s, split=" "))[1])

# library(magrittr)
# ll <- "a-b-c-d"
# strsplit(ll, "-") %>% sapply(extract2, 1)

# Keep cell lines that were actually screened and have RNA-Seq
cells_screened <- unique(as.vector(rspdata$CCLEName))
cells_screened[1:5]
usecells <- intersect(cells_screened, colnames(rna))
# rna <- dplyr::select(rna, ENSG, GeneName, usecells)
rna <- dplyr::select(rna, usecells)
message("Cells screend: ", length(cells_screened))
message("Cells screened and have RNA-Seq: ", length(usecells))

# Drop genes with all zero counts
# tmp = rna[rowSums(rna[,3:ncol(rna)])==0,]; sum(tmp[,3:ncol(tmp)])
# rna <- rna[rowSums(rna[,3:ncol(rna)])>0,]
rna <- rna[rowSums(rna)>0,]

# Seems like we don't have meta for one cell line --> what should we do??
# Thus, extract tissue from cell name
colnames(rna)[1:5]
cmeta$CCLEName[1:5]
message("Cells sequenced: ", ncol(rna))
message("Cells with metadata: ", nrow(cmeta))
message("Intersect btw rna cells and meta cells: ", length(intersect(colnames(rna), cmeta$CCLEName)))
message("Cell that doesn't appear in meta: ", setdiff(colnames(rna), cmeta$CCLEName))
tissuetype <- sapply(colnames(rna), function(s) paste(unlist(strsplit(s, "_"))[-1], collapse="_"))


# ==============================================================
#   Gene mappings
# ==============================================================
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
gene_mapping[1:3,]
sapply(gene_mapping, FUN=function(x) length(unique(x)))

idx <- !(duplicated(gene_mapping$ENSEMBL) | duplicated(gene_mapping$ENTREZID))
gene_mapping <- gene_mapping[idx,]
sapply(gene_mapping, FUN=function(x) length(unique(x)))

# Keep rna for only those genes that have unique mapping
# gene_subset <- intersect(rna$ENSG, gene_mapping$ENSEMBL)
tmp_genes <- intersect(rownames(rna), gene_mapping$ENSEMBL)
rna <- rna[tmp_genes,]



# ==============================================================
#   Select the subset of genes to proceed
# ==============================================================
if (rna_gene_set == "lincs") {
  # L1000
  l1k <- read.table(file.path(basedir, "../../data/raw/L1000.txt"), sep="\t", header=1)
  l1k <- dplyr::rename(l1k, "ENTREZID"="ID", "SymbolLINCS"="pr_gene_symbol")
  l1k <- dplyr::select(l1k, ENTREZID, SymbolLINCS)
  l1k <- l1k[order(l1k$ENTREZID),]
  l1k <- merge(l1k, gene_mapping, by="ENTREZID")
  l1k_ <- l1k[!(l1k$SymbolLINCS==l1k$SYMBOL),]  # Note, might be some problem with gene names (?!)
  tmp_genes <- intersect(rownames(rna), l1k$ENSEMBL)
  ldata <- rna[tmp_genes,]
  rna <- ldata
} else if (rna_gene_set == "top_exp") {
  # genes with highest expression (count) level
  n_top_genes <- 978
  x <- apply(rna, MARGIN=1, FUN=quantile, 0.25)
  y <- x[order(x, decreasing=T)][1:n_top_genes]
  rna <- rna[names(y),]
} else {
  # Keep the original
  rna <- rna
}



# ==============================================================
#   Create DESeqDataSet for normalization and make some plots
# ==============================================================
# Reorder the samples based on tissue type (for plotting)
idx <- c(order(tissuetype)); tissuetype <- tissuetype[idx]
table(tissuetype)
rna <- rna[,idx]

# Get a subset of samples
# set.seed(0)
# dfrna <- rna[,sample(ncol(rna), 20)]
# dfrna <- rna[,1:50]


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
df1[1:3,]

par(mfrow=c(1,1)); pdf(file.path(outdir, paste0(save_data_name, '_boxplot_log2(cnts).pdf')), width=100)
# par(mfrow=c(1,1)); pdf(file.path(outdir, 'boxplot_raw_log2(counts).pdf'), width=100)
# pdf(file.path(outdir, 'boxplot_raw_log2(counts)_facets.pdf'), width=20)
ggplot(df1, aes(x=ind, y=values)) +  
  # geom_violin(width=1.4) +
  geom_boxplot(aes(fill=factor(clrs)), alpha=0.9) +
  # geom_jitter(color="grey", size=0.7, alpha=0.5) +
  # scale_fill_viridis(discrete=T, option="magma") +
  ggtitle("CCLE samples (raw)") +
  xlab("") +
  ylab("log2(cnts+1)") +
  theme_dark() + 
  scale_colour_brewer(palette="Set1") +
  # theme(legend.position="bottom",
  #       legend.direction="horizontal",
  #       plot.title=element_text(size=11),
  #       axis.text.x=element_text(angle=60, hjust=1))
  theme(axis.text.x=element_text(angle=60, hjust=1))
# facet_wrap(~clrs)
dev.off()


# 
sort(colSums(rna)[1:3])/1e6


# --------------------------
# Create DESeqDataSet object
# --------------------------
# I don't create coldata from cellmeta because one sample is missing in the cellmeta
coldata <- as.data.frame(tissuetype)
dds <- DESeq2::DESeqDataSetFromMatrix(countData = rna,
                                      colData = coldata,
                                      # rowRanges = 
                                      design = ~1)  # design = ~batch + condition
print(dds)
is(dds) # type of object
class(dds)
slotNames(dds) # list of slot names
assayNames(dds)


# Subset the dataset (specific tissue types)
# dds <- subset(dds, select=colData(dds)$tissuetype %in% c("BONE", "KIDNEY"))


# ----------------------------------------------------
# Normalization for sequencing depth using sizeFactors
# ----------------------------------------------------
# Compute size factors
dds <- DESeq2::estimateSizeFactors(dds)
DESeq2::sizeFactors(dds)[1:3]  # access and print the size factors
plot(sizeFactors(dds), colSums(counts(dds)), cex=0.1)
abline(lm(colSums(counts(dds)) ~ sizeFactors(dds) + 0))

# Compute dfs
cnts <- DESeq2::counts(dds)
logcnts <- log2(DESeq2::counts(dds) + 1)  # log of raw counts
log_sf <- log2(DESeq2::counts(dds, normalized=TRUE) + 1)  # log of normalized counts using sizeFactors

# We can also get the counts normalized with sizeFactors and log using normTransform
lognorm <- normTransform(dds, f=log2, pc=1)
assay(lognorm)[1:2, 1:4]
all(log_sf == assay(lognorm))  # note that assay(lognorm) and lognormcnts are the same


# ------------------------
# Plots - sample vs sample
# ------------------------
# Raw counts
par(mfrow=c(1,1)); png(file.path(outdir, paste0(save_data_name, '_2_random_samples_cnts.png')))
plot(cnts[,1], cnts[,2], cex=0.1,
     xlab=colnames(cnts)[1], ylab=colnames(cnts)[2], main="Raw count")
abline(lm(cnts[,2] ~ cnts[,1] + 0))
dev.off()

# Log of raw counts
par(mfrow=c(1,1)); png(file.path(outdir, paste0(save_data_name, '_2_random_samples_logcnts.png')))
plot(logcnts[,1], logcnts[,2], cex=0.1,
     xlab=colnames(logcnts)[1], ylab=colnames(logcnts)[2], main="Log of raw count")
abline(lm(logcnts[,2] ~ logcnts[,1] + 0))
dev.off()

# Log of counts normalized using sizeFactors
par(mfrow=c(1,1)); png(file.path(outdir, paste0(save_data_name, '_2_random_samples_log_sf.png')))
plot(log_sf[,1], log_sf[,2], cex=0.1,
     xlab=colnames(log_sf)[1], ylab=colnames(log_sf)[2], main="Log of counts normalized using sizeFactors")
abline(lm(log_sf[,2] ~ log_sf[,1] + 0))
dev.off()

# --- Is there any difference? ---
# Plot the same sample counts: normalized using log2 vs normalized using SizeFactors
" We don't see much difference between log2 and log2 with sizeFactors "
logcnts[1:2, 1:4]
log_sf[1:2, 1:4]
plot(logcnts[,1], log_sf[,1], cex=0.1)
abline(lm(log_sf[,1] ~ logcnts[,1] + 0))


# --------------------------
# Boxplots of subset samples
# --------------------------
n_samples <- 10
# png(file.path(outdir, paste0(save_data_name, '_boxplot_log2(cnts)_vs_log_sf.png')))
# layout(matrix(c(1,2), nrow=1, ncol=2), respect=T)
par(mfrow=c(1,2))
boxplot(logcnts[,1:n_samples], main="log2(cnts+1)", cex=0.1) # not normalized
boxplot(log_sf[,1:n_samples], main="log2(sizeFactors)", cex=0.1) # normalized
# dev.off()
# my_boxplot <- function(df1, df2, lbl1, lbl2, filename) {
#   n_samples <- 10
#   png(filename)
#   layout(matrix(c(1,2), nrow=1, ncol=2), respect=T)
#   # par(mfrow=c(1,2)); 
#   boxplot(df1[,1:n_samples], main=lbl1, cex=0.1) # not normalized
#   boxplot(df2[,1:n_samples], main=lbl2, cex=0.1) # normalized  
#   dev.off()
# }
# my_boxplot(df1=logcnts, df2=log_sf, lbl1="log2(cnts+1)", lbl2="log2(sizeFactors)",
#            filename=file.path(outdir, paste0(save_data_name, '_boxplot_log2(cnts)_vs_log_sf.png')))



# ==============================================================
#   Stabilizing count variance - VSD
# ==============================================================
t0 <- Sys.time()
vsd <- varianceStabilizingTransformation(dds)
vsd_runtime <- Sys.time() - t0
glue("vsd run time: {vsd_runtime/60} mins")

# Get the actual vsd values
vsd_data <- assay(vsd)


# ---------------------------------
# Boxplots - compare normalizations
# ---------------------------------
n_samples <- 10
png(file.path(outdir, paste0(save_data_name, '_boxplot_compare_normalizations.png')))
layout(matrix(c(1,2,3), nrow=1, ncol=3), respect=T)
ylim <- c(0, 20)
boxplot(logcnts[,1:n_samples], main="log2(cnts+1)", cex=0.1, ylim=ylim)
boxplot(log_sf[,1:n_samples], main="log2(sizeFactors)", cex=0.1, ylim=ylim)
boxplot(vsd_data[,1:n_samples], main="vsd", cex=0.1, ylim=ylim)
dev.off()


# -----------------------------
# Plot - sample vs sample - vsd
# -----------------------------
par(mfrow=c(1,1)); png(file.path(outdir, paste0(save_data_name, '_2_random_samples_vsd.png')))
plot(vsd_data[,1], vsd_data[,2], cex=0.1, main="VSD")
abline(lm(vsd_data[,2] ~ vsd_data[,1] + 0))
dev.off()

# Plot (compare log2, sizeFactors, svd)
png(file.path(outdir, paste0(save_data_name, '_2_random_samples_comparison.png')))
layout(matrix(c(1,2,3), nrow=1, ncol=3), respect=T)
xlim <- c(-0.5, 20)
ylim <- c(-0.5, 20)  

plot(logcnts[,1], logcnts[,2], cex=0.1, main="log2(cts+1)",
     xlim=xlim, ylim=ylim, panel.first = grid())
abline(lm(logcnts[,2] ~ logcnts[,1] + 0))

plot(log_sf[,1], log_sf[,2], cex=0.1, main="log2(sizeFactors)",
     xlim=xlim, ylim=ylim, panel.first = grid())
abline(lm(log_sf[,2] ~ log_sf[,1] + 0))

plot(vsd_data[,1], vsd_data[,2], cex=0.1, main="VSD",
     xlim=xlim, ylim=ylim, panel.first = grid())
abline(lm(vsd_data[,2] ~ vsd_data[,1] + 0))
dev.off()


# ----------------
# Create tidy data
# ----------------
vsd_data_t <- as.data.frame(t(vsd_data))
vsd_data_t$CCLEName <- rownames(vsd_data_t)  # create col to merge on
vsd_data_t <- vsd_data_t[, c(ncol(vsd_data_t), seq(ncol(vsd_data_t)-1))]

vsd_cmeta <- as.data.frame(colData(vsd))
vsd_cmeta$CCLEName <- rownames(vsd_cmeta)  # create col to merge on
vsd_cmeta <- vsd_cmeta[, c(ncol(vsd_cmeta), seq(ncol(vsd_cmeta)-1))]

# rownames(rspdata) <- rspdata$CellName  # this won't work becuase rownames must be unique
# rownames(vsd_data_t) <- NULL 
dim(vsd_data_t)
dim(rspdata)
vsd_data_t[1:2, 1:3]
rspdata[1:2, 1:3]

# Merge data
# tidy_data_vsd <- merge(rspdata, vsd_data_t, by="CCLEName")
df1 <- merge(vsd_cmeta, vsd_data_t, by="CCLEName")
tidy_data_vsd <- merge(rspdata, df1, by="CCLEName")
tidy_data_vsd[1:3, 1:17]

# Save data
write.table(tidy_data_vsd, file.path(outdir, paste0("tidy_data_", save_data_name, "_ccle_vsd.txt")), sep="\t")
write.table(vsd_data, file.path(outdir, paste0(save_data_name, "_ccle_vsd.txt")), sep="\t")
write.table(vsd_cmeta, file.path(outdir, paste0(save_data_name, "_ccle_cmeta.txt")), sep="\t")


# ==============================================================
#   Stabilizing count variance - rlog
# ==============================================================
# t0 <- Sys.time()
# rld <- rlog(dds)
# rlog_runtime <- Sys.time() - t0
# # https://stackoverflow.com/questions/46085274/is-there-a-string-formatting-operator-in-r-similar-to-pythons
# glue("rlog run time: {rlog_runtime/60} mins")
# plot(assay(rld)[,1], assay(rld)[,2], cex=0.1)
# abline(lm(assay(rld)[,2] ~ assay(rld)[,1] + 0))


# Subset the DESeqTransform
# !!!SUPER USEFUL!!! --> that's what I tried to implement with Python
# sf_sub <- subset(lognorm, select=colData(lognorm)$tissuetype %in% c("BONE", "KIDNEY"))
# vsd_sub <- subset(vsd, select=colData(vsd)$tissuetype %in% c("BONE", "KIDNEY"))



# ==============================================================
#     Some EDA plots - TODO: not finished!!
# ==============================================================
# ---
# PCA
# ---
plotPCA(vsd, intgroup="tissuetype")
# pca_data <- plotPCA(lognorm, intgroup="tissuetype", returnData=T)
# pca_data <- plotPCA(vsd, intgroup="tissuetype", returnData=T)
# pca_data[1:3,]


# ----------
# Clustering
# ----------
par(mfrow=c(1, 2))
plot(hclust(dist(t(assay(vsd)))), labels=colData(vsd)$tissuetype)
plot(hclust(dist(t(assay(rld)))), labels=colData(rld)$tissuetype)


# -------
# Heatmap
# -------
# https://davetang.org/muse/2018/05/15/making-a-heatmap-in-r-with-the-pheatmap-package/
heatmap.2(as.matrix(assay(vsd)), 
          scale="row", 
          hclust=function(x) hclust(x, method="average"), 
          distfun=function(x) as.dist((1-cor(t(x)))/2), 
          trace="none", 
          density="none", 
          labRow="",
          cexCol=0.7)

topgenes <- head(rownames(resSort), 20)
mat <- assay(rld)[topgenes,]
mat <- mat - rowMeans(mat)
df <- as.data.frame(colData(dds)[,c("dex","cell")])
pheatmap(mat, annotation_col=df)

pheatmap(assay(vsd))

# SVA



# summary(rna[,1:4])
# stats.per.sample <- data.frame(t(do.call(cbind, lapply(rna, FUN = summary))))
# # Add some columns to the stats per sample
# stats.per.sample$libsum <- apply(rna, 2, sum)  # sum of the library
# stats.per.sample$perc05 <- apply(rna, 2, quantile, 0.05)
# stats.per.sample$perc10 <- apply(rna, 2, quantile, 0.10)
# stats.per.sample$perc90 <- apply(rna, 2, quantile, 0.90)
# stats.per.sample$perc95 <- apply(rna, 2, quantile, 0.95)
# stats.per.sample$zeros <- apply(rna==0, 2, sum)
# stats.per.sample$percent.zeros <- 100*stats.per.sample$zeros/nrow(rna)

# par(mfrow=c(3,1))
# hist(as.matrix(dfrna), col="blue", border="white", breaks=100)
# 
# hist(as.matrix(dfrna), col="blue", border="white",
#      breaks=20000, xlim=c(0,2000), main="Counts per gene",
#      xlab="Counts (truncated axis)", ylab="Number of genes", 
#      las=1, cex.axis=0.7)
# 
# epsilon <- 1 # pseudo-count to avoid problems with log(0)
# hist(as.matrix(log2(dfrna + epsilon)), breaks=100, col="blue", border="white",
#      main="Log2-transformed counts per gene", xlab="log2(counts+1)", ylab="Number of genes", 
#      las=1, cex.axis=0.7)



# =========================================================
# =========================================================
# Inconsistensy btw Broad and Ours data - fibroblast
# =========================================================
# =========================================================
rna <- read.table(file.path(datadir_molecular, "CCLE_DepMap_18q3_RNAseq_reads_20180718.gct.txt"),
                  sep="\t", skip=2, header=3, na.strings=c("NA", ""), check.names=F)
dim(rna)
colnames(rna)[1:4]
rna[1:2, 1:4]
rownames(rna) <- rna$Name
rna <- rna[,3:ncol(rna)]

rownames(rna) <- sapply(rownames(rna), FUN=function(s) unlist(strsplit(as.character(s), split=".", fixed=T))[1])
rna <- rna[order(rownames(rna)),]

colnames(rna) <- sapply(colnames(rna),
                        FUN=function(s) unlist(strsplit(s, split=" "))[1])
# -------------------------------------------------------
ll1 = c('HS229T', 'HS739T', 'HS840T', 'HS895T', 'RKN')
ll2 = c('COLO699', 'COV504', 'OC316')
# 'HS229T_LUNG'
# 'HS729_SOFT_TISSUE'
# 'HS840T_UPPER_AERODIGESTIVE_TRACT'
# 'HS895T_SKIN'
# 'RKN_OVARY'
func <- function(x) {
  for (i in ll1) {
    if (grepl(i, x))
      return(T)
  }
  return(F)
}
func('HS840T_UPPER_AERODIGESTIVE_TRACT')  

# Values in rna
colnames(rna)[sapply(colnames(rna), FUN=function(x) func(x))]

# Values in response
length(unique(rspdata$CCLEName))
unique(rspdata$CCLEName[sapply(rspdata$CCLEName, FUN=function(x) func(x))])

# Fibro
sum(sapply(cellmeta$Histology, FUN=function(x) grepl(tolower('fib'), tolower(x))))
# sum(sapply(cellmeta$HistSubtype1, FUN=function(x) grepl(tolower('fib'), tolower(x))))
cellmeta[sapply(cellmeta$Histology, FUN=function(x) grepl(tolower('fib'), tolower(x))),]

# RNA fibroblast
sum(sapply(colnames(rna), FUN=function(x) grepl(tolower('fib'), tolower(x))))
rna_fibro <- rna[,sapply(colnames(rna), FUN=function(x) grepl(tolower('fib'), tolower(x)))]
colnames(rna_fibro)

# Response fibroblast
sum(sapply(rspdata$CCLEName, FUN=function(x) grepl(tolower('fib'), tolower(x))))


