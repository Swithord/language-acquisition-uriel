#!/usr/bin/env Rscript

library(data.table)
library(ggplot2)
library(fixest)
library(broom)
library(glmnet)
library(Matrix)
library(patchwork)
library(scales)

set.seed(1234)

n_cores <- suppressWarnings(as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = "1")))
if (is.na(n_cores) || n_cores < 1L) n_cores <- 1L

data.table::setDTthreads(1L)
fixest::setFixest_nthreads(1L)

out_dir <- Sys.getenv("OUTPUT_DIR", unset = "artifacts/results_uriel")
data_dir <- Sys.getenv("DATA_DIR", unset = "data")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "plots"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "tables"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(out_dir, "rdata"), showWarnings = FALSE, recursive = TRUE)

stex  <- fread(file.path(data_dir, "stex_with_uriel_distances.csv"))
toefl <- fread(file.path(data_dir, "toefl_with_uriel_distances.csv"))

distance_vars <- c(
  "featural_dist",
  "script_dist",
  "geographic_dist",
  "genetic_dist",
  "syntactic_dist",
  "morphological_dist",
  "phonological_dist",
  "inventory_dist"
)

mechanistic_vars <- c(
  "syntactic_dist",
  "morphological_dist",
  "phonological_dist",
  "inventory_dist"
)

z_distance_vars <- paste0("z_", distance_vars)
z_mechanistic_vars <- paste0("z_", mechanistic_vars)

toefl_outcomes <- c("Reading", "Listening", "Speaking", "Writing", "Total")
stex_primary_outcomes <- c("Speaking", "new_feat", "new_sounds")
stex_secondary_gaussian <- c("morph", "lex")

glmnet_inner_folds_default <- suppressWarnings(as.integer(Sys.getenv("GLMNET_INNER_FOLDS", unset = "10")))
if (is.na(glmnet_inner_folds_default) || glmnet_inner_folds_default < 3L) {
  glmnet_inner_folds_default <- 10L
}

glmnet_nlambda_default <- suppressWarnings(as.integer(Sys.getenv("GLMNET_NLAMBDA", unset = "100")))
if (is.na(glmnet_nlambda_default) || glmnet_nlambda_default < 20L) {
  glmnet_nlambda_default <- 100L
}

standardize_cols <- function(dt, cols, prefix = "z_") {
  for (v in cols) {
    mu <- mean(dt[[v]], na.rm = TRUE)
    s  <- sd(dt[[v]], na.rm = TRUE)
    new_name <- paste0(prefix, v)
    if (is.na(s) || s == 0) {
      dt[, (new_name) := NA_real_]
    } else {
      dt[, (new_name) := (get(v) - mu) / s]
    }
  }
  invisible(dt)
}

rmse <- function(y, pred) sqrt(mean((y - pred)^2))
mae  <- function(y, pred) mean(abs(y - pred))

oos_r2 <- function(y, pred, train_mean) {
  denom <- sum((y - train_mean)^2)
  if (denom == 0) return(NA_real_)
  1 - sum((y - pred)^2) / denom
}

poisson_dev <- function(y, mu) {
  mu <- pmax(mu, 1e-8)
  term <- ifelse(y == 0, 0, y * log(y / mu))
  2 * mean(term - (y - mu))
}

mode_char <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA_character_)
  names(sort(table(x), decreasing = TRUE))[1]
}

save_plot <- function(p, filename, width = 10, height = 7, dpi = 300) {
  ggsave(
    filename = file.path(out_dir, "plots", filename),
    plot = p,
    width = width,
    height = height,
    dpi = dpi
  )
}

save_objects <- function(obj_names, filename) {
  save(
    list = obj_names,
    file = file.path(out_dir, "rdata", filename),
    envir = .GlobalEnv,
    compress = "xz"
  )
}

plapply <- function(X, FUN, ..., mc.cores = n_cores) {
  if (.Platform$OS.type == "unix" && mc.cores > 1L) {
    parallel::mclapply(X, FUN, ..., mc.cores = mc.cores)
  } else {
    lapply(X, FUN, ...)
  }
}

make_group_vfolds <- function(groups, v = 10, repeats = 5, seed = 1234) {
  groups <- as.character(groups)
  ug <- unique(groups)
  v_eff <- min(v, length(ug))
  folds <- list()
  idx <- 1L

  for (r in seq_len(repeats)) {
    set.seed(seed + r)
    ug_shuf <- sample(ug, length(ug), replace = FALSE)
    split_id <- rep(seq_len(v_eff), length.out = length(ug_shuf))

    for (k in seq_len(v_eff)) {
      test_groups <- ug_shuf[split_id == k]
      test_idx  <- which(groups %in% test_groups)
      train_idx <- setdiff(seq_along(groups), test_idx)

      if (length(test_idx) == 0L || length(train_idx) == 0L) next

      folds[[idx]] <- list(
        repeat_id = r,
        fold = k,
        train = train_idx,
        test = test_idx
      )
      idx <- idx + 1L
    }
  }

  folds
}

make_leave_one_group_folds <- function(groups) {
  groups <- as.character(groups)
  ug <- unique(groups)
  folds <- vector("list", length(ug))

  for (i in seq_along(ug)) {
    test_groups <- ug[i]
    test_idx  <- which(groups %in% test_groups)
    train_idx <- setdiff(seq_along(groups), test_idx)

    folds[[i]] <- list(
      repeat_id = 1L,
      fold = i,
      train = train_idx,
      test = test_idx
    )
  }

  folds
}

make_sparse_x <- function(data, rhs) {
  mm <- sparse.model.matrix(
    as.formula(paste0("~ ", rhs)),
    data = data
  )
  mm[, colnames(mm) != "(Intercept)", drop = FALSE]
}

tidy_lm <- function(fit, dataset, outcome, model_name, model_block) {
  td <- as.data.table(broom::tidy(fit, conf.int = TRUE))
  gl <- as.data.table(broom::glance(fit))
  td[, `:=`(
    dataset = dataset,
    outcome = outcome,
    model_name = model_name,
    model_block = model_block,
    n = nobs(fit),
    r.squared = gl$r.squared[1],
    adj.r.squared = gl$adj.r.squared[1],
    AIC = AIC(fit),
    BIC = BIC(fit)
  )]
  td
}

tidy_fixest <- function(fit, dataset, outcome, model_name, model_block, family_name) {
  td <- as.data.table(broom::tidy(fit, conf.int = TRUE))
  td[, `:=`(
    dataset = dataset,
    outcome = outcome,
    model_name = model_name,
    model_block = model_block,
    family_name = family_name,
    n = nobs(fit),
    AIC = suppressWarnings(tryCatch(AIC(fit), error = function(e) NA_real_)),
    BIC = suppressWarnings(tryCatch(BIC(fit), error = function(e) NA_real_))
  )]
  td
}

evaluate_glmnet_cv <- function(data, outcome, specs, folds, family = c("gaussian", "poisson")) {
  family <- match.arg(family)
  y <- data[[outcome]]

  x_cache <- vector("list", length(specs))
  names(x_cache) <- vapply(specs, function(s) s$name, character(1))

  for (i in seq_along(specs)) {
    if (!identical(specs[[i]]$type, "baseline_mean")) {
      x_cache[[i]] <- make_sparse_x(data, specs[[i]]$rhs)
    }
  }

  eval_one_spec <- function(spec, x_mat = NULL) {
    fold_res <- plapply(folds, function(fd) {
      y_train <- y[fd$train]
      y_test  <- y[fd$test]

      fallback_pred <- rep(mean(y_train), length(y_test))

      if (identical(spec$type, "baseline_mean")) {
        pred <- fallback_pred
      } else {
        x_train <- x_mat[fd$train, , drop = FALSE]
        x_test  <- x_mat[fd$test,  , drop = FALSE]

        inner_folds <- min(glmnet_inner_folds_default, length(y_train))
        if (inner_folds < 3L || ncol(x_train) == 0L || length(unique(y_train)) <= 1L) {
          pred <- fallback_pred
        } else {
          cvfit <- tryCatch(
            cv.glmnet(
              x = x_train,
              y = y_train,
              family = family,
              alpha = 0,
              nfolds = inner_folds,
              nlambda = glmnet_nlambda_default,
              standardize = TRUE,
              type.measure = if (family == "gaussian") "mse" else "deviance"
            ),
            error = function(e) NULL
          )

          if (is.null(cvfit)) {
            pred <- fallback_pred
          } else {
            pred <- as.numeric(
              predict(cvfit, newx = x_test, s = "lambda.1se", type = "response")
            )
          }
        }
      }

      data.table(
        model_name = spec$name,
        repeat_id = fd$repeat_id,
        fold = fd$fold,
        n_test = length(y_test),
        rmse = rmse(y_test, pred),
        mae = mae(y_test, pred),
        oos_r2 = if (family == "gaussian") oos_r2(y_test, pred, mean(y_train)) else NA_real_,
        poisson_dev = if (family == "poisson") poisson_dev(y_test, pred) else NA_real_
      )
    })

    rbindlist(fold_res, use.names = TRUE, fill = TRUE)
  }

  all_res <- vector("list", length(specs))

  for (i in seq_along(specs)) {
    spec <- specs[[i]]
    if (identical(spec$type, "baseline_mean")) {
      all_res[[i]] <- eval_one_spec(spec, x_mat = NULL)
    } else {
      all_res[[i]] <- eval_one_spec(spec, x_mat = x_cache[[spec$name]])
    }
  }

  rbindlist(all_res, use.names = TRUE, fill = TRUE)
}

coef_forest_plot <- function(dt, title_text) {
  plot_dt <- copy(dt)
  plot_dt <- plot_dt[
    term != "(Intercept)" &
      !grepl("^Sex|^C::|^L1_code::|^L2_code::|^Family", term)
  ]

  ggplot(plot_dt, aes(x = estimate, y = reorder(term, estimate))) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey50") +
    geom_point(size = 1.8) +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.15) +
    facet_wrap(~ outcome, scales = "free_y") +
    labs(
      title = title_text,
      x = "Estimate (95% CI)",
      y = NULL
    ) +
    theme_bw(base_size = 11)
}

perf_bar_plot <- function(dt, metric, title_text) {
  ggplot(dt, aes(x = reorder(model_name, get(metric)), y = get(metric))) +
    geom_col() +
    coord_flip() +
    facet_wrap(~ outcome, scales = "free_y") +
    labs(
      title = title_text,
      x = NULL,
      y = metric
    ) +
    theme_bw(base_size = 11)
}

stex[, `:=`(
  Sex = factor(Sex),
  C = factor(C),
  L1_code = factor(L1_code),
  L2_code = factor(L2_code),
  Family = factor(Family)
)]

toefl[, `:=`(
  L1_code = factor(L1_code),
  L2_code = factor(L2_code)
)]

standardize_cols(stex, distance_vars)
standardize_cols(toefl, distance_vars)

stex_main <- copy(
  stex[
    !is.na(L2_code) &
      L2 != "Monolingual"
  ]
)

stex_main <- stex_main[
  complete.cases(stex_main[, ..distance_vars])
]

stex_main[, pair_id := factor(paste(L1_code, L2_code, sep = "__"))]

stex_pair <- stex_main[, .(
  n = .N,
  Speaking = mean(Speaking, na.rm = TRUE),
  new_feat = mean(new_feat, na.rm = TRUE),
  new_sounds = mean(new_sounds, na.rm = TRUE),
  morph = mean(morph, na.rm = TRUE),
  lex = mean(lex, na.rm = TRUE),
  AaA = mean(AaA, na.rm = TRUE),
  LoR = mean(LoR, na.rm = TRUE),
  Edu.day = mean(Edu.day, na.rm = TRUE),
  Enroll = mean(Enroll, na.rm = TRUE),
  prop_female = mean(Sex == "Female", na.rm = TRUE),
  C = factor(mode_char(as.character(C))),
  L1_code = factor(as.character(first(L1_code))),
  L2_code = factor(as.character(first(L2_code))),
  featural_dist = first(featural_dist),
  script_dist = first(script_dist),
  geographic_dist = first(geographic_dist),
  genetic_dist = first(genetic_dist),
  syntactic_dist = first(syntactic_dist),
  morphological_dist = first(morphological_dist),
  phonological_dist = first(phonological_dist),
  inventory_dist = first(inventory_dist),
  z_featural_dist = first(z_featural_dist),
  z_script_dist = first(z_script_dist),
  z_geographic_dist = first(z_geographic_dist),
  z_genetic_dist = first(z_genetic_dist),
  z_syntactic_dist = first(z_syntactic_dist),
  z_morphological_dist = first(z_morphological_dist),
  z_phonological_dist = first(z_phonological_dist),
  z_inventory_dist = first(z_inventory_dist)
), by = pair_id]

toefl_cor <- rbindlist(lapply(toefl_outcomes, function(y) {
  rbindlist(lapply(distance_vars, function(d) {
    data.table(
      dataset = "toefl",
      outcome = y,
      distance = d,
      cor = cor(toefl[[y]], toefl[[d]], use = "pairwise.complete.obs")
    )
  }))
}))

fwrite(toefl_cor, file.path(out_dir, "tables", "01_toefl_correlations.csv"))

stex_pair_cor <- rbindlist(lapply(c("Speaking", "new_feat", "new_sounds"), function(y) {
  rbindlist(lapply(distance_vars, function(d) {
    data.table(
      dataset = "stex_pair",
      outcome = y,
      distance = d,
      cor = cor(stex_pair[[y]], stex_pair[[d]], use = "pairwise.complete.obs")
    )
  }))
}))

fwrite(stex_pair_cor, file.path(out_dir, "tables", "02_stex_pair_correlations.csv"))
fwrite(stex_pair, file.path(out_dir, "tables", "03_stex_pair_summary.csv"))

p_toefl_scatter <- ggplot(
  melt(
    copy(toefl),
    measure.vars = c("z_syntactic_dist", "z_morphological_dist", "z_phonological_dist", "z_inventory_dist"),
    variable.name = "distance",
    value.name = "z_distance"
  ),
  aes(x = z_distance, y = Total)
) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~ distance, scales = "free_x") +
  theme_bw(base_size = 11) +
  labs(
    title = "TOEFL Total vs selected URIEL distances",
    x = "Standardised URIEL distance",
    y = "TOEFL Total"
  )

save_plot(p_toefl_scatter, "01_toefl_total_scatter.png", width = 12, height = 7)

p_stex_pair_scatter <- ggplot(
  melt(
    copy(stex_pair),
    measure.vars = c("z_syntactic_dist", "z_morphological_dist", "z_phonological_dist", "z_inventory_dist"),
    variable.name = "distance",
    value.name = "z_distance"
  ),
  aes(x = z_distance, y = Speaking, size = n)
) +
  geom_point(alpha = 0.65) +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~ distance, scales = "free_x") +
  theme_bw(base_size = 11) +
  labs(
    title = "STEX pair-level mean Speaking vs selected URIEL distances",
    x = "Standardised URIEL distance",
    y = "Mean Speaking",
    size = "Pair size"
  )

save_plot(p_stex_pair_scatter, "02_stex_pair_speaking_scatter.png", width = 12, height = 7)

save_objects(
  c("toefl_cor", "stex_pair_cor", "stex_pair", "p_toefl_scatter", "p_stex_pair_scatter"),
  "01_descriptive_artifacts.RData"
)

run_toefl_models <- function(dat, outcome) {
  uni <- rbindlist(lapply(z_distance_vars, function(zv) {
    fit <- lm(as.formula(paste(outcome, "~", zv)), data = dat)
    tidy_lm(fit, dataset = "toefl", outcome = outcome, model_name = zv, model_block = "single_distance")
  }), use.names = TRUE, fill = TRUE)

  mech_rhs <- paste(z_mechanistic_vars, collapse = " + ")
  fit_mech <- lm(as.formula(paste(outcome, "~", mech_rhs)), data = dat)
  mech <- tidy_lm(fit_mech, dataset = "toefl", outcome = outcome, model_name = "mechanistic_block", model_block = "mechanistic_block")

  all_rhs <- paste(z_distance_vars, collapse = " + ")
  fit_all <- lm(as.formula(paste(outcome, "~", all_rhs)), data = dat)
  all8 <- tidy_lm(fit_all, dataset = "toefl", outcome = outcome, model_name = "all8_block", model_block = "all8_block")

  rbindlist(list(uni, mech, all8), use.names = TRUE, fill = TRUE)
}

toefl_inf <- rbindlist(lapply(toefl_outcomes, function(y) run_toefl_models(toefl, y)), use.names = TRUE, fill = TRUE)
toefl_inf[term != "(Intercept)", p_adj_fdr := p.adjust(p.value, method = "fdr"), by = .(outcome)]

fwrite(toefl_inf, file.path(out_dir, "tables", "04_toefl_inferential_results.csv"))

p_toefl_coef <- coef_forest_plot(
  toefl_inf[model_block == "single_distance" & grepl("^z_", term)],
  "TOEFL: one-distance-at-a-time inferential models"
)
save_plot(p_toefl_coef, "03_toefl_inferential_coefficients.png", width = 14, height = 9)

stex_base_covars <- c("AaA", "LoR", "Edu.day", "Sex", "Enroll")

run_stex_model <- function(dat, outcome, dist_var, family_type = c("gaussian", "negbin"), interaction = FALSE) {
  family_type <- match.arg(family_type)
  zdist <- paste0("z_", dist_var)

  rhs_terms <- c(stex_base_covars)
  if (!interaction) {
    rhs_terms <- c(zdist, rhs_terms)
    model_name <- paste0(outcome, "__", zdist)
    model_block <- "single_distance"
  } else {
    rhs_terms <- c(zdist, paste0(zdist, ":AaA"), paste0(zdist, ":LoR"), rhs_terms)
    model_name <- paste0(outcome, "__", zdist, "__interactions")
    model_block <- "interaction"
  }

  fml <- as.formula(
    paste0(
      outcome, " ~ ",
      paste(rhs_terms, collapse = " + "),
      " | C + L1_code + L2_code"
    )
  )

  if (family_type == "gaussian") {
    fit <- feols(fml, data = dat, cluster = ~pair_id)
    td <- tidy_fixest(fit, dataset = "stex", outcome = outcome, model_name = model_name, model_block = model_block, family_name = "gaussian")
  } else {
    fit <- fenegbin(fml, data = dat, cluster = ~pair_id)
    td <- tidy_fixest(fit, dataset = "stex", outcome = outcome, model_name = model_name, model_block = model_block, family_name = "negbin")
  }

  td
}

run_stex_block_model <- function(dat, outcome, block = c("mechanistic", "all8"), family_type = c("gaussian", "negbin")) {
  block <- match.arg(block)
  family_type <- match.arg(family_type)

  dist_terms <- if (block == "mechanistic") z_mechanistic_vars else z_distance_vars
  rhs_terms <- c(dist_terms, stex_base_covars)

  fml <- as.formula(
    paste0(
      outcome, " ~ ",
      paste(rhs_terms, collapse = " + "),
      " | C + L1_code + L2_code"
    )
  )

  if (family_type == "gaussian") {
    fit <- feols(fml, data = dat, cluster = ~pair_id)
    td <- tidy_fixest(fit, dataset = "stex", outcome = outcome, model_name = paste0(outcome, "__", block), model_block = paste0(block, "_block"), family_name = "gaussian")
  } else {
    fit <- fenegbin(fml, data = dat, cluster = ~pair_id)
    td <- tidy_fixest(fit, dataset = "stex", outcome = outcome, model_name = paste0(outcome, "__", block), model_block = paste0(block, "_block"), family_name = "negbin")
  }

  td
}

stex_inf_list <- list()

stex_inf_list[[length(stex_inf_list) + 1L]] <- rbindlist(
  lapply(distance_vars, function(d) run_stex_model(stex_main, "Speaking", d, family_type = "gaussian", interaction = FALSE)),
  use.names = TRUE, fill = TRUE
)

stex_inf_list[[length(stex_inf_list) + 1L]] <- run_stex_block_model(stex_main, "Speaking", block = "mechanistic", family_type = "gaussian")
stex_inf_list[[length(stex_inf_list) + 1L]] <- run_stex_block_model(stex_main, "Speaking", block = "all8", family_type = "gaussian")

stex_inf_list[[length(stex_inf_list) + 1L]] <- rbindlist(
  lapply(distance_vars, function(d) run_stex_model(stex_main, "Speaking", d, family_type = "gaussian", interaction = TRUE)),
  use.names = TRUE, fill = TRUE
)

for (y in c("new_feat", "new_sounds")) {
  dat_y <- copy(stex_main[!is.na(get(y))])

  stex_inf_list[[length(stex_inf_list) + 1L]] <- rbindlist(
    lapply(distance_vars, function(d) run_stex_model(dat_y, y, d, family_type = "negbin", interaction = FALSE)),
    use.names = TRUE, fill = TRUE
  )

  stex_inf_list[[length(stex_inf_list) + 1L]] <- run_stex_block_model(dat_y, y, block = "mechanistic", family_type = "negbin")
  stex_inf_list[[length(stex_inf_list) + 1L]] <- run_stex_block_model(dat_y, y, block = "all8", family_type = "negbin")
}

for (y in stex_secondary_gaussian) {
  dat_y <- copy(stex_main[!is.na(get(y))])
  if (nrow(dat_y) >= 200) {
    stex_inf_list[[length(stex_inf_list) + 1L]] <- rbindlist(
      lapply(distance_vars, function(d) run_stex_model(dat_y, y, d, family_type = "gaussian", interaction = FALSE)),
      use.names = TRUE, fill = TRUE
    )
    stex_inf_list[[length(stex_inf_list) + 1L]] <- run_stex_block_model(dat_y, y, block = "mechanistic", family_type = "gaussian")
    stex_inf_list[[length(stex_inf_list) + 1L]] <- run_stex_block_model(dat_y, y, block = "all8", family_type = "gaussian")
  }
}

stex_inf <- rbindlist(stex_inf_list, use.names = TRUE, fill = TRUE)
stex_inf[term != "(Intercept)", p_adj_fdr := p.adjust(p.value, method = "fdr"), by = .(outcome, model_block)]

fwrite(stex_inf, file.path(out_dir, "tables", "05_stex_inferential_results.csv"))

p_stex_coef <- coef_forest_plot(
  stex_inf[
    outcome %in% c("Speaking", "new_feat", "new_sounds") &
      model_block == "single_distance" &
      grepl("^z_", term)
  ],
  "STEX: one-distance-at-a-time inferential models"
)
save_plot(p_stex_coef, "04_stex_inferential_coefficients.png", width = 14, height = 10)

save_objects(
  c("toefl_inf", "stex_inf", "p_toefl_coef", "p_stex_coef"),
  "02_inferential_artifacts.RData"
)

run_stex_pair_sensitivity <- function(dat_pair, outcome) {
  uni <- rbindlist(lapply(z_distance_vars, function(zv) {
    fit <- feols(
      as.formula(
        paste0(
          outcome, " ~ ", zv,
          " + AaA + LoR + Edu.day + prop_female + Enroll | C + L1_code + L2_code"
        )
      ),
      data = dat_pair,
      weights = ~n
    )
    tidy_fixest(fit, dataset = "stex_pair", outcome = outcome, model_name = zv, model_block = "single_distance_pair", family_name = "gaussian")
  }), use.names = TRUE, fill = TRUE)

  mech_rhs <- paste(c(z_mechanistic_vars, "AaA", "LoR", "Edu.day", "prop_female", "Enroll"), collapse = " + ")
  fit_mech <- feols(
    as.formula(
      paste0(outcome, " ~ ", mech_rhs, " | C + L1_code + L2_code")
    ),
    data = dat_pair,
    weights = ~n
  )
  mech <- tidy_fixest(fit_mech, dataset = "stex_pair", outcome = outcome, model_name = "mechanistic_block", model_block = "mechanistic_pair", family_name = "gaussian")

  all_rhs <- paste(c(z_distance_vars, "AaA", "LoR", "Edu.day", "prop_female", "Enroll"), collapse = " + ")
  fit_all <- feols(
    as.formula(
      paste0(outcome, " ~ ", all_rhs, " | C + L1_code + L2_code")
    ),
    data = dat_pair,
    weights = ~n
  )
  all8 <- tidy_fixest(fit_all, dataset = "stex_pair", outcome = outcome, model_name = "all8_block", model_block = "all8_pair", family_name = "gaussian")

  rbindlist(list(uni, mech, all8), use.names = TRUE, fill = TRUE)
}

stex_pair_inf <- run_stex_pair_sensitivity(stex_pair, "Speaking")
fwrite(stex_pair_inf, file.path(out_dir, "tables", "06_stex_pair_sensitivity_results.csv"))

save_objects(
  c("stex_pair_inf"),
  "03_pair_sensitivity_artifacts.RData"
)

toefl_specs <- c(
  list(list(name = "baseline_mean", type = "baseline_mean", rhs = NULL)),
  lapply(z_distance_vars, function(zv) {
    list(name = paste0("ridge_", zv), type = "glmnet", rhs = zv)
  }),
  list(list(
    name = "ridge_mechanistic",
    type = "glmnet",
    rhs = paste(z_mechanistic_vars, collapse = " + ")
  )),
  list(list(
    name = "ridge_all8",
    type = "glmnet",
    rhs = paste(z_distance_vars, collapse = " + ")
  ))
)

toefl_pred_list <- list()

for (y in toefl_outcomes) {
  dat_y <- copy(toefl[complete.cases(toefl[, c(y, z_distance_vars), with = FALSE])])
  dat_y[, row_id := seq_len(.N)]
  folds <- make_leave_one_group_folds(dat_y$row_id)

  res <- evaluate_glmnet_cv(
    data = dat_y,
    outcome = y,
    specs = toefl_specs,
    folds = folds,
    family = "gaussian"
  )

  res[, `:=`(
    dataset = "toefl",
    outcome = y,
    scheme = "LOOCV"
  )]

  toefl_pred_list[[length(toefl_pred_list) + 1L]] <- res
  gc()
}

toefl_pred_fold <- rbindlist(toefl_pred_list, use.names = TRUE, fill = TRUE)
toefl_pred_summary <- toefl_pred_fold[, .(
  mean_rmse = mean(rmse, na.rm = TRUE),
  sd_rmse   = sd(rmse, na.rm = TRUE),
  mean_mae  = mean(mae, na.rm = TRUE),
  sd_mae    = sd(mae, na.rm = TRUE),
  mean_oos_r2 = mean(oos_r2, na.rm = TRUE),
  sd_oos_r2   = sd(oos_r2, na.rm = TRUE)
), by = .(dataset, outcome, scheme, model_name)]

fwrite(toefl_pred_fold,    file.path(out_dir, "tables", "07_toefl_predictive_fold_metrics.csv"))
fwrite(toefl_pred_summary, file.path(out_dir, "tables", "08_toefl_predictive_summary.csv"))

p_toefl_perf <- perf_bar_plot(
  toefl_pred_summary,
  metric = "mean_rmse",
  title_text = "TOEFL predictive performance (LOOCV; lower RMSE is better)"
)
save_plot(p_toefl_perf, "05_toefl_predictive_rmse.png", width = 14, height = 8)

stex_baseline_rhs <- "AaA + LoR + Edu.day + Sex + Enroll + C + L2 + Family"

stex_specs <- c(
  list(list(name = "baseline_mean", type = "baseline_mean", rhs = NULL)),
  list(list(name = "ridge_baseline", type = "glmnet", rhs = stex_baseline_rhs)),
  lapply(z_distance_vars, function(zv) {
    list(
      name = paste0("ridge_baseline_plus_", zv),
      type = "glmnet",
      rhs = paste(stex_baseline_rhs, zv, sep = " + ")
    )
  }),
  list(list(
    name = "ridge_baseline_plus_mechanistic",
    type = "glmnet",
    rhs = paste(stex_baseline_rhs, paste(z_mechanistic_vars, collapse = " + "), sep = " + ")
  )),
  list(list(
    name = "ridge_baseline_plus_all8",
    type = "glmnet",
    rhs = paste(stex_baseline_rhs, paste(z_distance_vars, collapse = " + "), sep = " + ")
  ))
)

run_stex_predictive <- function(dat, outcome, family = c("gaussian", "poisson")) {
  family <- match.arg(family)

  union_rhs <- paste(
    stex_baseline_rhs,
    paste(z_distance_vars, collapse = " + "),
    sep = " + "
  )

  vars_needed <- unique(all.vars(as.formula(paste0(outcome, " ~ ", union_rhs))))
  dat_use <- copy(dat[complete.cases(dat[, ..vars_needed])])

  schemes <- list(
    pair_id = make_group_vfolds(dat_use$pair_id, v = 10, repeats = 5, seed = 2024),
    L1_code = make_leave_one_group_folds(dat_use$L1_code),
    L2_code = make_leave_one_group_folds(dat_use$L2_code)
  )

  res_all <- list()

  for (scheme_name in names(schemes)) {
    res <- evaluate_glmnet_cv(
      data = dat_use,
      outcome = outcome,
      specs = stex_specs,
      folds = schemes[[scheme_name]],
      family = family
    )
    res[, `:=`(
      dataset = "stex",
      outcome = outcome,
      scheme = scheme_name
    )]
    res_all[[scheme_name]] <- res
    gc()
  }

  rbindlist(res_all, use.names = TRUE, fill = TRUE)
}

stex_pred_list <- list()

stex_pred_list[[length(stex_pred_list) + 1L]] <- run_stex_predictive(
  dat = stex_main,
  outcome = "Speaking",
  family = "gaussian"
)

for (y in c("new_feat", "new_sounds")) {
  dat_y <- copy(stex_main[!is.na(get(y))])
  stex_pred_list[[length(stex_pred_list) + 1L]] <- run_stex_predictive(
    dat = dat_y,
    outcome = y,
    family = "poisson"
  )
  gc()
}

stex_pred_fold <- rbindlist(stex_pred_list, use.names = TRUE, fill = TRUE)

stex_pred_summary <- stex_pred_fold[, .(
  mean_rmse = mean(rmse, na.rm = TRUE),
  sd_rmse   = sd(rmse, na.rm = TRUE),
  mean_mae  = mean(mae, na.rm = TRUE),
  sd_mae    = sd(mae, na.rm = TRUE),
  mean_oos_r2 = mean(oos_r2, na.rm = TRUE),
  sd_oos_r2   = sd(oos_r2, na.rm = TRUE),
  mean_poisson_dev = mean(poisson_dev, na.rm = TRUE),
  sd_poisson_dev   = sd(poisson_dev, na.rm = TRUE)
), by = .(dataset, outcome, scheme, model_name)]

fwrite(stex_pred_fold,    file.path(out_dir, "tables", "09_stex_predictive_fold_metrics.csv"))
fwrite(stex_pred_summary, file.path(out_dir, "tables", "10_stex_predictive_summary.csv"))

p_stex_perf_speaking <- perf_bar_plot(
  stex_pred_summary[outcome == "Speaking"],
  metric = "mean_rmse",
  title_text = "STEX Speaking predictive performance (lower RMSE is better)"
)
save_plot(p_stex_perf_speaking, "06_stex_speaking_predictive_rmse.png", width = 14, height = 8)

p_stex_perf_counts <- perf_bar_plot(
  stex_pred_summary[outcome %in% c("new_feat", "new_sounds")],
  metric = "mean_poisson_dev",
  title_text = "STEX count-outcome predictive performance (lower Poisson deviance is better)"
)
save_plot(p_stex_perf_counts, "07_stex_counts_predictive_deviance.png", width = 14, height = 8)

save_objects(
  c(
    "toefl_pred_fold",
    "toefl_pred_summary",
    "stex_pred_fold",
    "stex_pred_summary",
    "p_toefl_perf",
    "p_stex_perf_speaking",
    "p_stex_perf_counts"
  ),
  "04_predictive_artifacts.RData"
)

toefl_best_single <- toefl_inf[
  model_block == "single_distance" & term != "(Intercept)"
][order(outcome, p.value)][
  , .SD[1], by = outcome
]

fwrite(toefl_best_single, file.path(out_dir, "tables", "11_toefl_best_single_distance_terms.csv"))

stex_best_single <- stex_inf[
  outcome %in% c("Speaking", "new_feat", "new_sounds") &
    model_block == "single_distance" &
    term != "(Intercept)"
][order(outcome, p.value)][
  , .SD[1], by = outcome
]

fwrite(stex_best_single, file.path(out_dir, "tables", "12_stex_best_single_distance_terms.csv"))

toefl_best_pred <- toefl_pred_summary[order(outcome, mean_rmse)][, .SD[1], by = .(outcome, scheme)]
stex_best_pred  <- stex_pred_summary[
  order(outcome, scheme, fifelse(outcome == "Speaking", mean_rmse, mean_poisson_dev))
][
  ,
  .SD[1],
  by = .(outcome, scheme)
]

fwrite(toefl_best_pred, file.path(out_dir, "tables", "13_toefl_best_predictive_models.csv"))
fwrite(stex_best_pred,  file.path(out_dir, "tables", "14_stex_best_predictive_models.csv"))

save_objects(
  c("toefl_best_single", "stex_best_single", "toefl_best_pred", "stex_best_pred"),
  "05_summary_tables.RData"
)

save.image(file = file.path(out_dir, "rdata", "99_analysis_workspace.RData"), compress = "xz")

cat("\nanalysis complete.\n")
cat("outputs written to: ", normalizePath(out_dir), "\n\n")

cat("key files:\n")
cat("  - 04_toefl_inferential_results.csv\n")
cat("  - 05_stex_inferential_results.csv\n")
cat("  - 08_toefl_predictive_summary.csv\n")
cat("  - 10_stex_predictive_summary.csv\n")
cat("  - 11_toefl_best_single_distance_terms.csv\n")
cat("  - 12_stex_best_single_distance_terms.csv\n")
cat("  - 13_toefl_best_predictive_models.csv\n")
cat("  - 14_stex_best_predictive_models.csv\n")
cat("  - rdata/99_analysis_workspace.RData\n")