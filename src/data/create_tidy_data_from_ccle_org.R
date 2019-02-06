# TODO: rnaseq files (cnts, rpkm) contain two column for gene names: 1. "Name" (ENSG), 2. "Description" (gene name)
# we may need to use the gene names when mapping rna-seq data to combined!

# TODO: apply the following processing to the gene expression data
# https://dockflow.org/workflow/simple-single-cell/
# -----------------------------------------------------------------------------------------------------------------

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

# Choose what data to process and save in terms of rna-seq genes 
rna_gene_set <- "lincs";  save_data_name <- "lincs" # landmark genes
# rna_gene_set <- "top_exp_1k";  save_data_name <- "top_genes"  # most expressed genes ~1000
# rna_gene_set <- "full";  save_data_name <- "full_gene_set"  # full set of genes after filtering

# Set path to source file location
# Sessop --> Set Working Directory --> To Source File Location
# basedir <- "/Users/apartin/Dropbox/work/pilot1/cell-line"
basedir <- getwd()

# Utils
base::source("utils_data.R")

# Outdir
outdir <- file.path(basedir, "../../data/processed/from_ccle_org")
if (!dir.exists(outdir)) {dir.create(outdir)}
# list.files()

# Data folders
datadir <- "/Users/apartin/work/jdacs/cell-line-data/ccle/from_broad_institute"
datadir_molecular <- file.path(datadir, "current_data_11-08-2018")
datadir_cmeta <- file.path(datadir, "cell_line_annotations")
datadir_dmeta <- file.path(datadir, "pharmacological_profiling")

# Data filenames
cmeta_filename <- "CCLE_sample_info_file_2012-10-18.txt"
dmeta_filename <- "CCLE_NP24.2009_profiling_2012.02.20.csv"
rspdata_filename <- "CCLE_NP24.2009_Drug_data_2015.02.24.csv"
rnaseq_cnts_filename <- "CCLE_DepMap_18q3_RNAseq_reads_20180718.gct.txt"
rnaseq_rpkm_filename <- "CCLE_DepMap_18q3_RNAseq_RPKM_20180718.gct.txt"

# Data full path
cmeta_fullpath <- file.path(datadir_cmeta, cmeta_filename)
dmeta_fullpath <- file.path(datadir_dmeta, dmeta_filename)
rspdata_fullpath <- file.path(datadir_dmeta, rspdata_filename)
rnaseq_cnts_fullpath <- file.path(datadir_molecular, rnaseq_cnts_filename)
rnaseq_rpkm_fullpath <- file.path(datadir_molecular, rnaseq_rpkm_filename)



# ==============================================================
#   Load data
# ==============================================================
# Load cell metadata
cmeta <- load_cmeta(cmeta_fullpath)
# sapply(cmeta, FUN=function(x) length(unique(x)))
# apply(cmeta, MARGIN=2, FUN=function(x) length(unique(x)))

# Load drug metadata
dmeta <- load_dmeta(dmeta_fullpath) 
# apply(dmeta, MARGIN=2, FUN=function(x) length(unique(x)))

# Load response data
rspdata <- load_rspdata(rspdata_fullpath)
# apply(dplyr::select(rspdata, CCLEName, CellName, Drug), MARGIN=2,
#       FUN=function(x) length(unique(x)))
# colSums(is.na(rspdata))

# Load rna-seq and gene name mappings (ENSG, Name) as it appears in original df
ll <- load_rnaseq(rnaseq_fullpath=rnaseq_cnts_fullpath)
rna <- ll$rna
gene_names <- ll$gene_names
rm(ll)

# Problem with cmeta: it seems like we don't have cmeta for one cell line 
# that was screened and sequenced ... thus, extract tissue from cell line name
colnames(rna)[1:3]
cmeta$CCLEName[1:3]
message("Cells with metadata: ", nrow(cmeta))
message("Intersect btw rna cells and meta cells: ", length(intersect(colnames(rna), cmeta$CCLEName)))
message("Cell that doesn't appear in meta: ", setdiff(colnames(rna), cmeta$CCLEName))
tissuetype <- sapply(colnames(rna), function(s) paste(unlist(strsplit(s, "_"))[-1], collapse="_"))

# Load gene name mappings
gene_mapping <- get_gene_mappgins()
sapply(gene_mapping, FUN=function(x) length(unique(x)))

# Keep rna for only those genes that have unique name mappings
# gene_subset <- intersect(rna$ENSG, gene_mapping$ENSEMBL)
tmp_genes <- intersect(rownames(rna), gene_mapping$ENSEMBL)
rna <- rna[tmp_genes,]

# Select the subset of genes to proceed
# TODO: get_gene_subset() uses gene_mappings to get unique genes. This leaves
# us with 965 instead of 978 genes. We may want to use gene_names instead.
##rna <- get_gene_subset(rna=rna, rna_gene_set="lincs", gene_mapping=gene_mapping,
##                       l1k_dir=file.path(basedir, "../../data/raw/L1000.txt"))


# ==============================================================
#   Load data RPKM data
# ==============================================================
# Load rpkm rna-seq and keep the genes from cnts df
ll <- load_rnaseq(rnaseq_fullpath=rnaseq_rpkm_fullpath)
rna_rpkm <- ll$rna
gene_names_rpkm <- ll$gene_names
rm(ll)

rna_rpkm <- rna_rpkm[rownames(rna),]



# ==============================================================
#   Create DESeqDataSet for normalization and make some plots
# ==============================================================
# ----------------------
# Boxplots - all samples
# ----------------------
# Reorder the samples based on tissue type (for boxplots)
idx <- c(order(tissuetype))
tissuetype <- tissuetype[idx]
table(tissuetype)
rna <- rna[,idx]
rna_rpkm <- rna_rpkm[,idx]

# Boxplot of multiple samples - cnts
plot_boxplots(rna=rna, tissuetype=tissuetype,
              filename=paste0(save_data_name, "_boxplots_log2(cnts)"), n=NULL)

# Boxplot of multiple samples - rpkm
plot_boxplots(rna=rna_rpkm, tissuetype=tissuetype,
              filename=paste0(save_data_name, "_boxplots_rpkm"), n=NULL)

sort(colSums(rna)[1:3])/1e6


# --------------------------
# Create DESeqDataSet object
# --------------------------
# Don't create coldata from cellmeta because one sample is missing in the cellmeta
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
DESeq2::sizeFactors(dds)[1:3]  # access and print size factors
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


# --------------------------
# Boxplots of subset samples
# --------------------------
n_samples <- 10
# png(file.path(outdir, paste0(save_data_name, '_boxplot_log2(cnts)_vs_log_sf.png')))
# layout(matrix(c(1,2), nrow=1, ncol=2), respect=T)
par(mfrow=c(1,2))
boxplot(logcnts[,1:n_samples], main="log2(cnts+1)", cex=0.1, xaxt="n") # not normalized
boxplot(log_sf[,1:n_samples], main="log2(sizeFactors)", cex=0.1, xaxt="n") # normalized



# ==============================================================
#   Stabilizing count variance - VSD
# ==============================================================
t0 <- Sys.time()
vsd <- varianceStabilizingTransformation(dds)
vsd_runtime <- Sys.time() - t0
glue("vsd runtime: {vsd_runtime} mins")

# Get the actual vsd values
vsd_data <- as.data.frame(assay(vsd))


# ----------------------
# Boxplots - all samples
# ----------------------
# Boxplot of multiple samples - vsd
plot_boxplots(rna=vsd_data, tissuetype=tissuetype,
              filename=paste0(save_data_name, "_boxplots_vsd"), n=NULL)


# ----------------------------------------
# Boxplots - compare normalizations (save)
# ----------------------------------------
n_samples <- 10
pdf(file.path(outdir, paste0(save_data_name, '_boxplot_compare_normalizations.pdf')))
layout(matrix(c(1,2,3,4), nrow=2, ncol=2), respect=T)
ylim <- c(0, 20)
boxplot(cnts[,1:n_samples], main="cnts", cex=0.1, xaxt="n")
boxplot(logcnts[,1:n_samples], main="log2(cnts+1)", cex=0.1, ylim=ylim, xaxt="n")
boxplot(log_sf[,1:n_samples], main="log2(sizeFactors)", cex=0.1, ylim=ylim, xaxt="n")
boxplot(vsd_data[,1:n_samples], main="vsd", cex=0.1, ylim=ylim, xaxt="n")
dev.off()


# ------------------------------------
# Plot - sample vs sample - vsd (save)
# ------------------------------------
# Plot (compare raw counts, log2, sizeFactors, svd)
pdf(file.path(outdir, paste0(save_data_name, '_sample_vs_sample_comparison.pdf')))
layout(matrix(c(1,2,3,4), nrow=2, ncol=2), respect=T)
xlim <- c(-0.5, 20)
ylim <- c(-0.5, 20)  

plot(cnts[,1], cnts[,2], cex=0.1, main="Raw count",
     panel.first=grid(), xaxt="n")
abline(lm(cnts[,2] ~ cnts[,1] + 0))

plot(logcnts[,1], logcnts[,2], cex=0.1, main="log2(cnts+1)",
     xlim=xlim, ylim=ylim, panel.first=grid(), xaxt="n")
abline(lm(logcnts[,2] ~ logcnts[,1] + 0))

plot(log_sf[,1], log_sf[,2], cex=0.1, main="log2(sizeFactors)",
     xlim=xlim, ylim=ylim, panel.first=grid(), xaxt="n")
abline(lm(log_sf[,2] ~ log_sf[,1] + 0))

plot(vsd_data[,1], vsd_data[,2], cex=0.1, main="VSD",
     xlim=xlim, ylim=ylim, panel.first=grid(), xaxt="n")
abline(lm(vsd_data[,2] ~ vsd_data[,1] + 0))
dev.off()



# ==============================================================
#   Create tidy data
# ==============================================================
# Keep cell lines that were actually screened and sequenced
cells_screened <- unique(as.vector(rspdata$CCLEName))
cells_screened[1:3]
usecells <- intersect(cells_screened, colnames(rna))
message("Cells sequenced: ", ncol(rna))
message("Cells screened: ", length(cells_screened))
message("Cells screened and sequenced: ", length(usecells))
rna <- dplyr::select(rna, usecells)


# Transpose rna and add col to merging with rspdata
transpose_rna <- function(df) {
  df_t <- as.data.frame(t(df))
  df_t$CCLEName <- rownames(df_t)  # create col to merge on
  df_t <- df_t[, c(ncol(df_t), seq(ncol(df_t)-1))] # put CCLEName as the first col
  return(df_t)
}
rna_rpkm_t <- transpose_rna(df=rna_rpkm)
vsd_data_t <- transpose_rna(df=vsd_data)

# Preproc cmeta for merging
vsd_cmeta <- as.data.frame(colData(vsd))
vsd_cmeta$CCLEName <- rownames(vsd_cmeta)  # create col to merge on
vsd_cmeta <- vsd_cmeta[, c(ncol(vsd_cmeta), seq(ncol(vsd_cmeta)-1))]

# rownames(rspdata) <- rspdata$CellName  # this won't work becuase rownames must be unique
# rownames(vsd_data_t) <- NULL 
dim(vsd_data_t)
dim(rspdata)
vsd_data_t[1:2, 1:3]
rspdata[1:2, 1:3]

# Merge expression and rspdata
merge_data <- function(rspdata, df, vsd_cmeta) {
  df1 <- merge(vsd_cmeta, df, by="CCLEName")
  tidy_data <- merge(rspdata, df1, by="CCLEName")
  return(tidy_data)
}
tidy_data_vsd <- merge_data(rspdata=rspdata, df=vsd_data_t, vsd_cmeta=vsd_cmeta)
tidy_data_rpkm <- merge_data(rspdata=rspdata, df=rna_rpkm_t, vsd_cmeta=vsd_cmeta)

# Add features prefix
rna_prfx <- "cell_rna."
colnames(tidy_data_vsd) <- sapply(colnames(tidy_data_vsd), FUN=function(s) dplyr::if_else(grepl("ENSG", s), true=paste0(rna_prfx, s), false=s))
colnames(tidy_data_rpkm) <- sapply(colnames(tidy_data_rpkm), FUN=function(s) dplyr::if_else(grepl("ENSG", s), true=paste0(rna_prfx, s), false=s))

# Save data
write.table(tidy_data_rpkm, file.path(outdir, paste0("tidy_data_ccle_rpkm_", save_data_name, ".txt")), sep="\t")
write.table(rna_rpkm_t, file.path(outdir, paste0("ccle_rpkm_", save_data_name, ".txt")), sep="\t")

write.table(tidy_data_vsd, file.path(outdir, paste0("tidy_data_ccle_vsd_", save_data_name, ".txt")), sep="\t")
write.table(vsd_data_t, file.path(outdir, paste0("ccle_vsd_", save_data_name, ".txt")), sep="\t")

write.table(vsd_cmeta, file.path(outdir, paste0("ccle_cmeta.txt")), sep="\t")



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


